#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::cuda::DeviceArrayF32Triplet;
use crate::indicators::kdj::{KdjBatchRange, KdjParams};
use crate::indicators::willr::build_willr_gpu_tables;
use cust::context::{CacheConfig, Context};
use cust::device::Device;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{
    mem_get_info, AsyncCopyDestination, CopyDestination, DeviceBuffer, LockedBuffer,
};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CudaKdjError {
    #[error("CUDA error: {0}")]
    Cuda(#[from] cust::error::CudaError),
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
    #[error("device mismatch: buf={buf}, current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
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

#[derive(Clone, Copy, Debug)]
pub struct CudaKdjPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaKdjPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

pub struct CudaKdj {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaKdjPolicy,
}

impl CudaKdj {
    pub fn new(device_id: usize) -> Result<Self, CudaKdjError> {
        Self::new_with_policy(device_id, CudaKdjPolicy::default())
    }

    pub fn new_with_policy(device_id: usize, policy: CudaKdjPolicy) -> Result<Self, CudaKdjError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/kdj_kernel.ptx"));
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
            _context: context,
            device_id: device_id as u32,
            policy,
        })
    }

    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }

    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaKdjError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free {
                Ok(())
            } else {
                Err(CudaKdjError::OutOfMemory {
                    required: required_bytes,
                    free,
                    headroom: headroom_bytes,
                })
            }
        } else {
            Ok(())
        }
    }

    fn ma_to_code(s: &str) -> Result<i32, CudaKdjError> {
        if s.eq_ignore_ascii_case("sma") {
            return Ok(0);
        } else if s.eq_ignore_ascii_case("ema") {
            return Ok(1);
        } else {
            return Err(CudaKdjError::InvalidInput(format!(
                "unsupported MA type '{}'; supported: sma, ema",
                s
            )));
        }
    }

    fn expand_grid(range: &KdjBatchRange) -> Result<Vec<KdjParams>, CudaKdjError> {
        fn axis_usize(a: (usize, usize, usize)) -> Result<Vec<usize>, CudaKdjError> {
            let (start, end, step) = a;
            if step == 0 || start == end {
                return Ok(vec![start]);
            }
            let mut v = Vec::new();
            if start < end {
                let mut cur = start;
                while cur <= end {
                    v.push(cur);
                    let next = cur.saturating_add(step);
                    if next == cur {
                        break;
                    }
                    cur = next;
                }
            } else {
                let mut cur = start;
                while cur >= end {
                    v.push(cur);
                    let next = cur.saturating_sub(step);
                    if next == cur {
                        break;
                    }
                    cur = next;
                    if cur == 0 && end > 0 {
                        break;
                    }
                }
            }
            if v.is_empty() {
                return Err(CudaKdjError::InvalidInput(format!(
                    "invalid range: start={}, end={}, step={}",
                    start, end, step
                )));
            }
            Ok(v)
        }
        fn axis_str(a: (String, String, String)) -> Vec<String> {
            let (start, end, _step) = a;
            if start == end {
                vec![start]
            } else {
                vec![start, end]
            }
        }
        let fks = axis_usize(range.fast_k_period)?;
        let sks = axis_usize(range.slow_k_period)?;
        let kmas = axis_str(range.slow_k_ma_type.clone());
        let sds = axis_usize(range.slow_d_period)?;
        let dmas = axis_str(range.slow_d_ma_type.clone());
        let mut out = Vec::new();
        for &fk in &fks {
            for &sk in &sks {
                for kma in &kmas {
                    for &sd in &sds {
                        for dma in &dmas {
                            out.push(KdjParams {
                                fast_k_period: Some(fk),
                                slow_k_period: Some(sk),
                                slow_k_ma_type: Some(kma.clone()),
                                slow_d_period: Some(sd),
                                slow_d_ma_type: Some(dma.clone()),
                            });
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    pub fn kdj_batch_dev(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        close_f32: &[f32],
        sweep: &KdjBatchRange,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32, DeviceArrayF32), CudaKdjError> {
        let len = high_f32.len();
        if len == 0 || low_f32.len() != len || close_f32.len() != len {
            return Err(CudaKdjError::InvalidInput(
                "input slices are empty or mismatched".into(),
            ));
        }

        let first_valid = (0..len)
            .find(|&i| {
                high_f32[i].is_finite() && low_f32[i].is_finite() && close_f32[i].is_finite()
            })
            .ok_or_else(|| CudaKdjError::InvalidInput("all values are NaN".into()))?
            as i32;

        let combos = Self::expand_grid(sweep)?;

        let mut fk: Vec<i32> = Vec::with_capacity(combos.len());
        let mut sk: Vec<i32> = Vec::with_capacity(combos.len());
        let mut sd: Vec<i32> = Vec::with_capacity(combos.len());
        let mut kma: Vec<i32> = Vec::with_capacity(combos.len());
        let mut dma: Vec<i32> = Vec::with_capacity(combos.len());
        let mut max_fk = 0usize;
        for p in &combos {
            let fkv = p.fast_k_period.unwrap_or(0);
            let skv = p.slow_k_period.unwrap_or(0);
            let sdv = p.slow_d_period.unwrap_or(0);
            if fkv == 0 || skv == 0 || sdv == 0 {
                return Err(CudaKdjError::InvalidInput(
                    "periods must be positive".into(),
                ));
            }
            fk.push(fkv as i32);
            sk.push(skv as i32);
            sd.push(sdv as i32);
            kma.push(Self::ma_to_code(
                p.slow_k_ma_type.as_deref().unwrap_or("sma"),
            )?);
            dma.push(Self::ma_to_code(
                p.slow_d_ma_type.as_deref().unwrap_or("sma"),
            )?);
            max_fk = max_fk.max(fkv);
        }
        let valid_tail = len as i32 - first_valid;
        if valid_tail < max_fk as i32 {
            return Err(CudaKdjError::InvalidInput(format!(
                "not enough valid data: need >= {}, have {}",
                max_fk, valid_tail
            )));
        }

        let tables = build_willr_gpu_tables(high_f32, low_f32);
        let level_count = tables.level_offsets.len() as i32;

        let nrows = combos.len();
        let sz_f32 = std::mem::size_of::<f32>();
        let sz_i32 = std::mem::size_of::<i32>();
        let bytes_inputs = close_f32
            .len()
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaKdjError::InvalidInput("size overflow (inputs)".into()))?;
        let tables_i32 = tables
            .log2
            .len()
            .checked_add(tables.level_offsets.len())
            .and_then(|e| e.checked_add(tables.nan_psum.len()))
            .ok_or_else(|| CudaKdjError::InvalidInput("size overflow (tables i32)".into()))?;
        let bytes_tables_i32 = tables_i32
            .checked_mul(sz_i32)
            .ok_or_else(|| CudaKdjError::InvalidInput("size overflow (tables i32 bytes)".into()))?;
        let tables_f32 = tables
            .st_max
            .len()
            .checked_add(tables.st_min.len())
            .ok_or_else(|| CudaKdjError::InvalidInput("size overflow (tables f32)".into()))?;
        let bytes_tables_f32 = tables_f32
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaKdjError::InvalidInput("size overflow (tables f32 bytes)".into()))?;
        let bytes_tables = bytes_tables_i32
            .checked_add(bytes_tables_f32)
            .ok_or_else(|| CudaKdjError::InvalidInput("size overflow (tables total)".into()))?;
        let params_len = fk
            .len()
            .checked_add(sk.len())
            .and_then(|e| e.checked_add(sd.len()))
            .and_then(|e| e.checked_add(kma.len()))
            .and_then(|e| e.checked_add(dma.len()))
            .ok_or_else(|| CudaKdjError::InvalidInput("size overflow (params)".into()))?;
        let bytes_params = params_len
            .checked_mul(sz_i32)
            .ok_or_else(|| CudaKdjError::InvalidInput("size overflow (params bytes)".into()))?;
        let rows_cols = nrows
            .checked_mul(len)
            .ok_or_else(|| CudaKdjError::InvalidInput("size overflow (rows*len)".into()))?;
        let bytes_outputs = rows_cols
            .checked_mul(3)
            .and_then(|e| e.checked_mul(sz_f32))
            .ok_or_else(|| CudaKdjError::InvalidInput("size overflow (outputs bytes)".into()))?;
        let required = bytes_inputs
            .checked_add(bytes_tables)
            .and_then(|e| e.checked_add(bytes_params))
            .and_then(|e| e.checked_add(bytes_outputs))
            .ok_or_else(|| CudaKdjError::InvalidInput("size overflow (required bytes)".into()))?;
        let headroom = 64 * 1024 * 1024;
        Self::will_fit(required, headroom)?;

        let combos_per_launch = nrows;

        let use_async = required > (64 * 1024 * 1024);

        let (d_close, d_log2, d_offsets, d_st_max, d_st_min, d_nan_psum) = if use_async {
            let h_close = LockedBuffer::from_slice(close_f32).map_err(CudaKdjError::Cuda)?;
            let h_log2 = LockedBuffer::from_slice(&tables.log2).map_err(CudaKdjError::Cuda)?;
            let h_offs =
                LockedBuffer::from_slice(&tables.level_offsets).map_err(CudaKdjError::Cuda)?;
            let h_max = LockedBuffer::from_slice(&tables.st_max).map_err(CudaKdjError::Cuda)?;
            let h_min = LockedBuffer::from_slice(&tables.st_min).map_err(CudaKdjError::Cuda)?;
            let h_nps = LockedBuffer::from_slice(&tables.nan_psum).map_err(CudaKdjError::Cuda)?;
            let mut dc = unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream) }
                .map_err(CudaKdjError::Cuda)?;
            let mut dl = unsafe {
                DeviceBuffer::<i32>::uninitialized_async(tables.log2.len(), &self.stream)
            }
            .map_err(CudaKdjError::Cuda)?;
            let mut dof = unsafe {
                DeviceBuffer::<i32>::uninitialized_async(tables.level_offsets.len(), &self.stream)
            }
            .map_err(CudaKdjError::Cuda)?;
            let mut dmx = unsafe {
                DeviceBuffer::<f32>::uninitialized_async(tables.st_max.len(), &self.stream)
            }
            .map_err(CudaKdjError::Cuda)?;
            let mut dmn = unsafe {
                DeviceBuffer::<f32>::uninitialized_async(tables.st_min.len(), &self.stream)
            }
            .map_err(CudaKdjError::Cuda)?;
            let mut dnp = unsafe {
                DeviceBuffer::<i32>::uninitialized_async(tables.nan_psum.len(), &self.stream)
            }
            .map_err(CudaKdjError::Cuda)?;
            unsafe { dc.async_copy_from(&h_close, &self.stream) }.map_err(CudaKdjError::Cuda)?;
            unsafe { dl.async_copy_from(&h_log2, &self.stream) }.map_err(CudaKdjError::Cuda)?;
            unsafe { dof.async_copy_from(&h_offs, &self.stream) }.map_err(CudaKdjError::Cuda)?;
            unsafe { dmx.async_copy_from(&h_max, &self.stream) }.map_err(CudaKdjError::Cuda)?;
            unsafe { dmn.async_copy_from(&h_min, &self.stream) }.map_err(CudaKdjError::Cuda)?;
            unsafe { dnp.async_copy_from(&h_nps, &self.stream) }.map_err(CudaKdjError::Cuda)?;
            (dc, dl, dof, dmx, dmn, dnp)
        } else {
            (
                DeviceBuffer::from_slice(close_f32).map_err(CudaKdjError::Cuda)?,
                DeviceBuffer::from_slice(&tables.log2).map_err(CudaKdjError::Cuda)?,
                DeviceBuffer::from_slice(&tables.level_offsets).map_err(CudaKdjError::Cuda)?,
                DeviceBuffer::from_slice(&tables.st_max).map_err(CudaKdjError::Cuda)?,
                DeviceBuffer::from_slice(&tables.st_min).map_err(CudaKdjError::Cuda)?,
                DeviceBuffer::from_slice(&tables.nan_psum).map_err(CudaKdjError::Cuda)?,
            )
        };

        let d_fk_all = DeviceBuffer::from_slice(&fk).map_err(CudaKdjError::Cuda)?;
        let d_sk_all = DeviceBuffer::from_slice(&sk).map_err(CudaKdjError::Cuda)?;
        let d_sd_all = DeviceBuffer::from_slice(&sd).map_err(CudaKdjError::Cuda)?;
        let d_km_all = DeviceBuffer::from_slice(&kma).map_err(CudaKdjError::Cuda)?;
        let d_dm_all = DeviceBuffer::from_slice(&dma).map_err(CudaKdjError::Cuda)?;

        let mut d_k =
            unsafe { DeviceBuffer::<f32>::uninitialized(rows_cols) }.map_err(CudaKdjError::Cuda)?;
        let mut d_d =
            unsafe { DeviceBuffer::<f32>::uninitialized(rows_cols) }.map_err(CudaKdjError::Cuda)?;
        let mut d_j =
            unsafe { DeviceBuffer::<f32>::uninitialized(rows_cols) }.map_err(CudaKdjError::Cuda)?;

        let mut func: Function = self.module.get_function("kdj_batch_f32").map_err(|_| {
            CudaKdjError::MissingKernelSymbol {
                name: "kdj_batch_f32",
            }
        })?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);

        let mut row0 = 0usize;
        while row0 < nrows {
            let rows = (nrows - row0)
                .min(combos_per_launch)
                .min(2_147_483_647usize);
            let grid_x = rows as u32;
            let grid: GridSize = (grid_x, 1, 1).into();
            let mut block_x = match self.policy.batch {
                BatchKernelPolicy::Plain { block_x } => block_x,
                _ => 256,
            };
            if block_x < 32 {
                block_x = 32;
            }
            let block: BlockSize = (block_x, 1, 1).into();
            if block_x > 1024 || grid_x == 0 {
                return Err(CudaKdjError::LaunchConfigTooLarge {
                    gx: grid_x,
                    gy: 1,
                    gz: 1,
                    bx: block_x,
                    by: 1,
                    bz: 1,
                });
            }

            unsafe {
                let mut close_ptr = d_close.as_device_ptr().as_raw();
                let mut log2_ptr = d_log2.as_device_ptr().as_raw();
                let mut offs_ptr = d_offsets.as_device_ptr().as_raw();
                let mut stmax_ptr = d_st_max.as_device_ptr().as_raw();
                let mut stmin_ptr = d_st_min.as_device_ptr().as_raw();
                let mut nanp_ptr = d_nan_psum.as_device_ptr().as_raw();

                let mut high_ptr = close_ptr;
                let mut low_ptr = close_ptr;

                let mut fk_ptr = d_fk_all.as_device_ptr().offset(row0 as isize).as_raw();
                let mut sk_ptr = d_sk_all.as_device_ptr().offset(row0 as isize).as_raw();
                let mut sd_ptr = d_sd_all.as_device_ptr().offset(row0 as isize).as_raw();
                let mut kma_ptr = d_km_all.as_device_ptr().offset(row0 as isize).as_raw();
                let mut dma_ptr = d_dm_all.as_device_ptr().offset(row0 as isize).as_raw();
                let mut series_len_i = len as i32;
                let mut first_i = first_valid as i32;
                let mut level_cnt_i = level_count as i32;
                let mut nrows_i = rows as i32;
                let mut outk_ptr =
                    unsafe { d_k.as_device_ptr().offset((row0 * len) as isize).as_raw() };
                let mut outd_ptr =
                    unsafe { d_d.as_device_ptr().offset((row0 * len) as isize).as_raw() };
                let mut outj_ptr =
                    unsafe { d_j.as_device_ptr().offset((row0 * len) as isize).as_raw() };

                let args: &mut [*mut c_void] = &mut [
                    &mut high_ptr as *mut _ as *mut c_void,
                    &mut low_ptr as *mut _ as *mut c_void,
                    &mut close_ptr as *mut _ as *mut c_void,
                    &mut log2_ptr as *mut _ as *mut c_void,
                    &mut offs_ptr as *mut _ as *mut c_void,
                    &mut stmax_ptr as *mut _ as *mut c_void,
                    &mut stmin_ptr as *mut _ as *mut c_void,
                    &mut nanp_ptr as *mut _ as *mut c_void,
                    &mut fk_ptr as *mut _ as *mut c_void,
                    &mut sk_ptr as *mut _ as *mut c_void,
                    &mut sd_ptr as *mut _ as *mut c_void,
                    &mut kma_ptr as *mut _ as *mut c_void,
                    &mut dma_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut level_cnt_i as *mut _ as *mut c_void,
                    &mut nrows_i as *mut _ as *mut c_void,
                    &mut outk_ptr as *mut _ as *mut c_void,
                    &mut outd_ptr as *mut _ as *mut c_void,
                    &mut outj_ptr as *mut _ as *mut c_void,
                ];

                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(CudaKdjError::Cuda)?;
            }
            row0 += rows;
        }

        self.stream.synchronize().map_err(CudaKdjError::Cuda)?;

        Ok((
            DeviceArrayF32 {
                buf: d_k,
                rows: nrows,
                cols: len,
            },
            DeviceArrayF32 {
                buf: d_d,
                rows: nrows,
                cols: len,
            },
            DeviceArrayF32 {
                buf: d_j,
                rows: nrows,
                cols: len,
            },
        ))
    }

    pub fn kdj_many_series_one_param_time_major_dev(
        &self,
        high_tm_f32: &[f32],
        low_tm_f32: &[f32],
        close_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &KdjParams,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32, DeviceArrayF32), CudaKdjError> {
        if cols == 0 || rows == 0 {
            return Err(CudaKdjError::InvalidInput(
                "series dims must be positive".into(),
            ));
        }
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaKdjError::InvalidInput("size overflow (dims)".into()))?;
        if high_tm_f32.len() != elems || low_tm_f32.len() != elems || close_tm_f32.len() != elems {
            return Err(CudaKdjError::InvalidInput(
                "time-major slices mismatch dims".into(),
            ));
        }
        let fk = params.fast_k_period.unwrap_or(9);
        let sk = params.slow_k_period.unwrap_or(3);
        let sd = params.slow_d_period.unwrap_or(3);
        let kma = Self::ma_to_code(params.slow_k_ma_type.as_deref().unwrap_or("sma"))?;
        let dma = Self::ma_to_code(params.slow_d_ma_type.as_deref().unwrap_or("sma"))?;

        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let idx = t * cols + s;
                if high_tm_f32[idx].is_finite()
                    && low_tm_f32[idx].is_finite()
                    && close_tm_f32[idx].is_finite()
                {
                    fv = Some(t as i32);
                    break;
                }
            }
            let f =
                fv.ok_or_else(|| CudaKdjError::InvalidInput(format!("series {} all NaN", s)))?;
            if rows - (f as usize) < fk {
                return Err(CudaKdjError::InvalidInput(format!(
                    "series {} insufficient data for fk {}",
                    s, fk
                )));
            }
            first_valids[s] = f;
        }

        let sz_f32 = std::mem::size_of::<f32>();
        let sz_i32 = std::mem::size_of::<i32>();
        let in_bytes = elems
            .checked_mul(3)
            .and_then(|e| e.checked_mul(sz_f32))
            .ok_or_else(|| CudaKdjError::InvalidInput("size overflow (inputs)".into()))?;
        let aux_bytes = cols
            .checked_mul(sz_i32)
            .ok_or_else(|| CudaKdjError::InvalidInput("size overflow (aux)".into()))?;
        let out_bytes = elems
            .checked_mul(3)
            .and_then(|e| e.checked_mul(sz_f32))
            .ok_or_else(|| CudaKdjError::InvalidInput("size overflow (outputs)".into()))?;
        let required = in_bytes
            .checked_add(aux_bytes)
            .and_then(|e| e.checked_add(out_bytes))
            .ok_or_else(|| CudaKdjError::InvalidInput("size overflow (required bytes)".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        let d_h = DeviceBuffer::from_slice(high_tm_f32).map_err(CudaKdjError::Cuda)?;
        let d_l = DeviceBuffer::from_slice(low_tm_f32).map_err(CudaKdjError::Cuda)?;
        let d_c = DeviceBuffer::from_slice(close_tm_f32).map_err(CudaKdjError::Cuda)?;
        let d_fv = DeviceBuffer::from_slice(&first_valids).map_err(CudaKdjError::Cuda)?;

        let mut d_k =
            unsafe { DeviceBuffer::<f32>::uninitialized(elems) }.map_err(CudaKdjError::Cuda)?;
        let mut d_d =
            unsafe { DeviceBuffer::<f32>::uninitialized(elems) }.map_err(CudaKdjError::Cuda)?;
        let mut d_j =
            unsafe { DeviceBuffer::<f32>::uninitialized(elems) }.map_err(CudaKdjError::Cuda)?;

        let mut func: Function = self
            .module
            .get_function("kdj_many_series_one_param_f32")
            .map_err(|_| CudaKdjError::MissingKernelSymbol {
                name: "kdj_many_series_one_param_f32",
            })?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);
        let mut block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            _ => 128,
        };
        if block_x < 32 {
            block_x = 32;
        }
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        if block_x > 1024 || grid_x == 0 {
            return Err(CudaKdjError::LaunchConfigTooLarge {
                gx: grid_x,
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }

        unsafe {
            let mut h_ptr = d_h.as_device_ptr().as_raw();
            let mut l_ptr = d_l.as_device_ptr().as_raw();
            let mut c_ptr = d_c.as_device_ptr().as_raw();
            let mut fv_ptr = d_fv.as_device_ptr().as_raw();
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut fk_i = fk as i32;
            let mut sk_i = sk as i32;
            let mut sd_i = sd as i32;
            let mut kma_i = kma as i32;
            let mut dma_i = dma as i32;
            let mut ko_ptr = d_k.as_device_ptr().as_raw();
            let mut do_ptr = d_d.as_device_ptr().as_raw();
            let mut jo_ptr = d_j.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut h_ptr as *mut _ as *mut c_void,
                &mut l_ptr as *mut _ as *mut c_void,
                &mut c_ptr as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut fk_i as *mut _ as *mut c_void,
                &mut sk_i as *mut _ as *mut c_void,
                &mut sd_i as *mut _ as *mut c_void,
                &mut kma_i as *mut _ as *mut c_void,
                &mut dma_i as *mut _ as *mut c_void,
                &mut ko_ptr as *mut _ as *mut c_void,
                &mut do_ptr as *mut _ as *mut c_void,
                &mut jo_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaKdjError::Cuda)?;
        }

        self.stream.synchronize().map_err(CudaKdjError::Cuda)?;

        Ok((
            DeviceArrayF32 {
                buf: d_k,
                rows,
                cols,
            },
            DeviceArrayF32 {
                buf: d_d,
                rows,
                cols,
            },
            DeviceArrayF32 {
                buf: d_j,
                rows,
                cols,
            },
        ))
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::kdj::KdjBatchRange;

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;

    fn synth_hlc_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if !v.is_finite() {
                continue;
            }
            let x = i as f32 * 0.0023;
            let off = (0.0029 * x.sin()).abs() + 0.1;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = 1 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = 3 * ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct KdjBatchDeviceState {
        cuda: CudaKdj,
        func: Function<'static>,
        d_close: DeviceBuffer<f32>,
        d_log2: DeviceBuffer<i32>,
        d_offsets: DeviceBuffer<i32>,
        d_st_max: DeviceBuffer<f32>,
        d_st_min: DeviceBuffer<f32>,
        d_nan_psum: DeviceBuffer<i32>,
        d_fk: DeviceBuffer<i32>,
        d_sk: DeviceBuffer<i32>,
        d_sd: DeviceBuffer<i32>,
        d_kma: DeviceBuffer<i32>,
        d_dma: DeviceBuffer<i32>,
        d_k: DeviceBuffer<f32>,
        d_d: DeviceBuffer<f32>,
        d_j: DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        level_count: i32,
        rows: usize,
        block_x: u32,
    }
    impl CudaBenchState for KdjBatchDeviceState {
        fn launch(&mut self) {
            let combos_per_launch = 65_535usize;
            let mut row0 = 0usize;
            while row0 < self.rows {
                let rows = (self.rows - row0).min(combos_per_launch);
                let grid: GridSize = (rows as u32, 1, 1).into();
                let block: BlockSize = (self.block_x, 1, 1).into();
                unsafe {
                    let mut close_ptr = self.d_close.as_device_ptr().as_raw();
                    let mut high_ptr = close_ptr;
                    let mut low_ptr = close_ptr;
                    let mut log2_ptr = self.d_log2.as_device_ptr().as_raw();
                    let mut offs_ptr = self.d_offsets.as_device_ptr().as_raw();
                    let mut stmax_ptr = self.d_st_max.as_device_ptr().as_raw();
                    let mut stmin_ptr = self.d_st_min.as_device_ptr().as_raw();
                    let mut nanp_ptr = self.d_nan_psum.as_device_ptr().as_raw();
                    let mut fk_ptr = self.d_fk.as_device_ptr().offset(row0 as isize).as_raw();
                    let mut sk_ptr = self.d_sk.as_device_ptr().offset(row0 as isize).as_raw();
                    let mut sd_ptr = self.d_sd.as_device_ptr().offset(row0 as isize).as_raw();
                    let mut kma_ptr = self.d_kma.as_device_ptr().offset(row0 as isize).as_raw();
                    let mut dma_ptr = self.d_dma.as_device_ptr().offset(row0 as isize).as_raw();
                    let mut series_len_i = self.len as i32;
                    let mut first_i = self.first_valid as i32;
                    let mut level_cnt_i = self.level_count;
                    let mut nrows_i = rows as i32;
                    let mut outk_ptr = self
                        .d_k
                        .as_device_ptr()
                        .offset((row0 * self.len) as isize)
                        .as_raw();
                    let mut outd_ptr = self
                        .d_d
                        .as_device_ptr()
                        .offset((row0 * self.len) as isize)
                        .as_raw();
                    let mut outj_ptr = self
                        .d_j
                        .as_device_ptr()
                        .offset((row0 * self.len) as isize)
                        .as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut high_ptr as *mut _ as *mut c_void,
                        &mut low_ptr as *mut _ as *mut c_void,
                        &mut close_ptr as *mut _ as *mut c_void,
                        &mut log2_ptr as *mut _ as *mut c_void,
                        &mut offs_ptr as *mut _ as *mut c_void,
                        &mut stmax_ptr as *mut _ as *mut c_void,
                        &mut stmin_ptr as *mut _ as *mut c_void,
                        &mut nanp_ptr as *mut _ as *mut c_void,
                        &mut fk_ptr as *mut _ as *mut c_void,
                        &mut sk_ptr as *mut _ as *mut c_void,
                        &mut sd_ptr as *mut _ as *mut c_void,
                        &mut kma_ptr as *mut _ as *mut c_void,
                        &mut dma_ptr as *mut _ as *mut c_void,
                        &mut series_len_i as *mut _ as *mut c_void,
                        &mut first_i as *mut _ as *mut c_void,
                        &mut level_cnt_i as *mut _ as *mut c_void,
                        &mut nrows_i as *mut _ as *mut c_void,
                        &mut outk_ptr as *mut _ as *mut c_void,
                        &mut outd_ptr as *mut _ as *mut c_void,
                        &mut outj_ptr as *mut _ as *mut c_void,
                    ];
                    self.cuda
                        .stream
                        .launch(&self.func, grid, block, 0, args)
                        .expect("kdj batch launch");
                }
                row0 += rows;
            }
            self.cuda.stream.synchronize().expect("kdj batch sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaKdj::new(0).expect("cuda kdj");
        let close = gen_series(ONE_SERIES_LEN);
        let (high, low) = synth_hlc_from_close(&close);
        let first_valid = (0..ONE_SERIES_LEN)
            .find(|&i| high[i].is_finite() && low[i].is_finite() && close[i].is_finite())
            .unwrap_or(0);

        let sweep = KdjBatchRange {
            fast_k_period: (9, 9 + PARAM_SWEEP - 1, 1),
            slow_k_period: (3, 3, 0),
            slow_k_ma_type: ("sma".into(), "sma".into(), "".into()),
            slow_d_period: (3, 3, 0),
            slow_d_ma_type: ("sma".into(), "sma".into(), "".into()),
        };
        let combos = CudaKdj::expand_grid(&sweep).expect("expand_grid");
        let rows = combos.len();
        let mut fk: Vec<i32> = Vec::with_capacity(rows);
        let mut sk: Vec<i32> = Vec::with_capacity(rows);
        let mut sd: Vec<i32> = Vec::with_capacity(rows);
        let mut kma: Vec<i32> = Vec::with_capacity(rows);
        let mut dma: Vec<i32> = Vec::with_capacity(rows);
        for p in &combos {
            fk.push(p.fast_k_period.unwrap_or(0) as i32);
            sk.push(p.slow_k_period.unwrap_or(0) as i32);
            sd.push(p.slow_d_period.unwrap_or(0) as i32);
            kma.push(CudaKdj::ma_to_code(p.slow_k_ma_type.as_deref().unwrap_or("sma")).unwrap());
            dma.push(CudaKdj::ma_to_code(p.slow_d_ma_type.as_deref().unwrap_or("sma")).unwrap());
        }

        let tables = build_willr_gpu_tables(&high, &low);
        let level_count = tables.level_offsets.len() as i32;

        let d_close: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::from_slice_async(&close, &cuda.stream) }.expect("d_close");
        let d_log2 = DeviceBuffer::from_slice(&tables.log2).expect("d_log2");
        let d_offsets = DeviceBuffer::from_slice(&tables.level_offsets).expect("d_offsets");
        let d_st_max = DeviceBuffer::from_slice(&tables.st_max).expect("d_st_max");
        let d_st_min = DeviceBuffer::from_slice(&tables.st_min).expect("d_st_min");
        let d_nan_psum = DeviceBuffer::from_slice(&tables.nan_psum).expect("d_nan_psum");

        let d_fk = DeviceBuffer::from_slice(&fk).expect("d_fk");
        let d_sk = DeviceBuffer::from_slice(&sk).expect("d_sk");
        let d_sd = DeviceBuffer::from_slice(&sd).expect("d_sd");
        let d_kma = DeviceBuffer::from_slice(&kma).expect("d_kma");
        let d_dma = DeviceBuffer::from_slice(&dma).expect("d_dma");

        let rows_cols = rows * ONE_SERIES_LEN;
        let d_k: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(rows_cols) }.expect("d_k");
        let d_d: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(rows_cols) }.expect("d_d");
        let d_j: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(rows_cols) }.expect("d_j");

        let func = cuda
            .module
            .get_function("kdj_batch_f32")
            .expect("kdj_batch_f32");
        let mut func: Function<'static> = unsafe { std::mem::transmute(func) };
        let _ = func.set_cache_config(CacheConfig::PreferL1);

        cuda.stream.synchronize().expect("sync after prep");
        Box::new(KdjBatchDeviceState {
            cuda,
            func,
            d_close,
            d_log2,
            d_offsets,
            d_st_max,
            d_st_min,
            d_nan_psum,
            d_fk,
            d_sk,
            d_sd,
            d_kma,
            d_dma,
            d_k,
            d_d,
            d_j,
            len: ONE_SERIES_LEN,
            first_valid,
            level_count,
            rows,
            block_x: 256,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "kdj",
            "one_series_many_params",
            "kdj_cuda_batch_dev",
            "1m_x_250",
            prep_one_series_many_params,
        )
        .with_mem_required(bytes_one_series_many_params())
        .with_sample_size(10)]
    }
}
