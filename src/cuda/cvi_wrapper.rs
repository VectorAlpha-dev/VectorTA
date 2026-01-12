#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::cvi::CviBatchRange;
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;

#[derive(thiserror::Error, Debug)]
pub enum CudaCviError {
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
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaCviPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaCviPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

pub struct CudaCvi {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaCviPolicy,
}

impl CudaCvi {
    pub fn new(device_id: usize) -> Result<Self, CudaCviError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/cvi_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
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
            policy: CudaCviPolicy::default(),
        })
    }

    #[inline]
    pub fn ctx(&self) -> Arc<Context> {
        self.context.clone()
    }
    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }
    #[inline]
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    pub fn set_policy(&mut self, policy: CudaCviPolicy) {
        self.policy = policy;
    }
    pub fn synchronize(&self) -> Result<(), CudaCviError> {
        self.stream.synchronize().map_err(Into::into)
    }

    fn first_valid_hl(high: &[f32], low: &[f32]) -> Result<usize, CudaCviError> {
        if high.is_empty() || low.is_empty() {
            return Err(CudaCviError::InvalidInput("empty input".into()));
        }
        let n = high.len().min(low.len());
        for i in 0..n {
            if !high[i].is_nan() && !low[i].is_nan() {
                return Ok(i);
            }
        }
        Err(CudaCviError::InvalidInput("all values are NaN".into()))
    }

    fn will_fit(bytes: usize, headroom: usize) -> bool {
        let check = match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        };
        if !check {
            return true;
        }
        if let Ok((free, _)) = mem_get_info() {
            bytes.saturating_add(headroom) <= free
        } else {
            true
        }
    }

    #[inline(always)]
    fn grid_1d_for_elems(elems: usize, block_x: u32) -> (u32, GridSize) {
        let gx = ((elems as u32) + block_x - 1) / block_x;
        let gx_clamped = gx.max(1);
        (gx_clamped, (gx_clamped, 1, 1).into())
    }

    #[inline(always)]
    fn bytes_required_batch(n_combos: usize, len: usize) -> Option<usize> {
        let f32b = std::mem::size_of::<f32>();
        let i32b = std::mem::size_of::<i32>();
        let in_bytes = (2usize).checked_mul(len)?.checked_mul(f32b)?;
        let params_each = (2usize).checked_mul(i32b)?.checked_add(f32b)?;
        let params_bytes = n_combos.checked_mul(params_each)?;
        let out_bytes = n_combos.checked_mul(len)?.checked_mul(f32b)?;
        in_bytes.checked_add(params_bytes)?.checked_add(out_bytes)
    }

    fn chunk_size_for_batch(n_combos: usize, len: usize) -> usize {
        let in_bytes = 2 * len * std::mem::size_of::<f32>();
        let params_bytes = n_combos * (std::mem::size_of::<i32>() * 2 + std::mem::size_of::<f32>());
        let out_per_combo = len * std::mem::size_of::<f32>();
        let headroom = 64 * 1024 * 1024;
        let mut chunk = n_combos.max(1);
        while chunk > 1 {
            let need = in_bytes + params_bytes + chunk * out_per_combo + headroom;
            if Self::will_fit(need, 0) {
                break;
            }
            chunk = (chunk + 1) / 2;
        }
        chunk.max(1)
    }

    fn validate_launch(
        &self,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    ) -> Result<(), CudaCviError> {
        let dev = Device::get_device(self.device_id)?;
        let max_gx = dev.get_attribute(DeviceAttribute::MaxGridDimX)? as u32;
        let max_gy = dev.get_attribute(DeviceAttribute::MaxGridDimY)? as u32;
        let max_gz = dev.get_attribute(DeviceAttribute::MaxGridDimZ)? as u32;
        let max_bx = dev.get_attribute(DeviceAttribute::MaxBlockDimX)? as u32;
        let max_by = dev.get_attribute(DeviceAttribute::MaxBlockDimY)? as u32;
        let max_bz = dev.get_attribute(DeviceAttribute::MaxBlockDimZ)? as u32;
        if gx > max_gx || gy > max_gy || gz > max_gz || bx > max_bx || by > max_by || bz > max_bz {
            return Err(CudaCviError::LaunchConfigTooLarge {
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

    pub fn cvi_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        sweep: &CviBatchRange,
    ) -> Result<DeviceArrayF32, CudaCviError> {
        if high.len() != low.len() {
            return Err(CudaCviError::InvalidInput("input length mismatch".into()));
        }
        let len = high.len();
        if len == 0 {
            return Err(CudaCviError::InvalidInput("empty input".into()));
        }
        let first_valid = Self::first_valid_hl(high, low)?;

        let (ps, pe, pst) = sweep.period;
        if ps == 0 {
            return Err(CudaCviError::InvalidInput("period must be > 0".into()));
        }
        let periods: Vec<usize> = if pst == 0 || ps == pe {
            vec![ps]
        } else if ps < pe {
            (ps..=pe).step_by(pst).collect()
        } else {
            let mut v = Vec::new();
            let s = pst.max(1);
            let mut cur = ps;
            loop {
                v.push(cur);
                if cur <= pe {
                    break;
                }
                let next = cur.saturating_sub(s);
                if next == cur {
                    break;
                }
                cur = next;
            }
            v.retain(|&x| x >= pe);
            v
        };
        if periods.is_empty() {
            return Err(CudaCviError::InvalidInput(format!(
                "invalid period range: start={} end={} step={}",
                ps, pe, pst
            )));
        }
        for &p in &periods {
            if p == 0 || p > len || (len - first_valid) < (2 * p - 1) {
                return Err(CudaCviError::InvalidInput(format!(
                    "invalid period {} for data length {} (valid after {}: {}), need >= {}",
                    p,
                    len,
                    first_valid,
                    len - first_valid,
                    2 * p - 1
                )));
            }
        }
        let n_combos = periods.len();

        let headroom = 64 * 1024 * 1024;
        let need = Self::bytes_required_batch(n_combos, len)
            .ok_or_else(|| CudaCviError::InvalidInput("size overflow".into()))?;
        if !Self::will_fit(need, headroom) {
            let (free, _) = mem_get_info().unwrap_or((0, 0));
            return Err(CudaCviError::OutOfMemory {
                required: need,
                free,
                headroom,
            });
        }

        let h_periods: Vec<i32> = periods.iter().map(|&p| p as i32).collect();
        let h_alphas: Vec<f32> = periods
            .iter()
            .map(|&p| (2.0f32 / (p as f32 + 1.0f32)))
            .collect();
        let h_warms: Vec<i32> = periods
            .iter()
            .map(|&p| (first_valid + (2 * p - 1)) as i32)
            .collect();

        let mut d_high_opt = Some(DeviceBuffer::from_slice(high)?);
        let mut d_low_opt = Some(DeviceBuffer::from_slice(low)?);
        let d_periods = DeviceBuffer::from_slice(&h_periods)?;
        let d_alphas = DeviceBuffer::from_slice(&h_alphas)?;
        let d_warms = DeviceBuffer::from_slice(&h_warms)?;

        let has_cvi_from_range = self.module.get_function("cvi_batch_from_range_f32").is_ok();
        let has_range_kernel = self.module.get_function("range_from_high_low_f32").is_ok();
        let mut d_range_opt: Option<DeviceBuffer<f32>> = None;
        if has_cvi_from_range {
            if has_range_kernel {
                let mut d_range: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }?;
                let range_func = self
                    .module
                    .get_function("range_from_high_low_f32")
                    .map_err(|_| CudaCviError::MissingKernelSymbol {
                        name: "range_from_high_low_f32",
                    })?;
                unsafe {
                    let mut len_i = len as i32;
                    let mut high_ptr = d_high_opt.as_ref().unwrap().as_device_ptr().as_raw();
                    let mut low_ptr = d_low_opt.as_ref().unwrap().as_device_ptr().as_raw();
                    let mut out_ptr = d_range.as_device_ptr().as_raw();
                    let block_x_range = 256u32;
                    let (gx_range, grid) = Self::grid_1d_for_elems(len, block_x_range);
                    let block: BlockSize = (block_x_range, 1, 1).into();
                    self.validate_launch(gx_range, 1, 1, block_x_range, 1, 1)?;
                    let args: &mut [*mut c_void] = &mut [
                        &mut high_ptr as *mut _ as *mut c_void,
                        &mut low_ptr as *mut _ as *mut c_void,
                        &mut len_i as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    self.stream.launch(&range_func, grid, block, 0, args)?;
                }

                d_high_opt = None;
                d_low_opt = None;
                d_range_opt = Some(d_range);
            } else {
                let mut r = vec![0f32; len];
                for i in 0..len {
                    r[i] = high[i] - low[i];
                }
                let dev = DeviceBuffer::from_slice(&r)?;
                d_range_opt = Some(dev);

                d_high_opt = None;
                d_low_opt = None;
            }
        }

        let total = n_combos
            .checked_mul(len)
            .ok_or_else(|| CudaCviError::InvalidInput("n_combos*len overflow".into()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }?;

        let func = if has_cvi_from_range {
            self.module
                .get_function("cvi_batch_from_range_f32")
                .map_err(|_| CudaCviError::MissingKernelSymbol {
                    name: "cvi_batch_from_range_f32",
                })?
        } else {
            self.module.get_function("cvi_batch_f32").map_err(|_| {
                CudaCviError::MissingKernelSymbol {
                    name: "cvi_batch_f32",
                }
            })?
        };

        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            BatchKernelPolicy::Auto => 256,
        };
        let chunk = {
            let max_blocks: usize = 16_384;
            (n_combos).min(max_blocks * (block_x as usize))
        };

        let mut launched = 0usize;
        while launched < n_combos {
            let cur = (n_combos - launched).min(chunk);
            let (gx, grid): (u32, GridSize) = Self::grid_1d_for_elems(cur, block_x);
            let block: BlockSize = (block_x, 1, 1).into();
            self.validate_launch(gx, 1, 1, block_x, 1, 1)?;
            unsafe {
                let mut len_i = len as i32;
                let mut first_i = first_valid as i32;
                let mut cur_i = cur as i32;
                let mut periods_ptr = d_periods
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * std::mem::size_of::<i32>()) as u64);
                let mut alphas_ptr = d_alphas
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * std::mem::size_of::<f32>()) as u64);
                let mut warms_ptr = d_warms
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * std::mem::size_of::<i32>()) as u64);
                let mut out_ptr = d_out
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * len * std::mem::size_of::<f32>()) as u64);
                if has_cvi_from_range {
                    let dr = d_range_opt.as_ref().expect("range device buffer missing");
                    let mut range_ptr = dr.as_device_ptr().as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut range_ptr as *mut _ as *mut c_void,
                        &mut periods_ptr as *mut _ as *mut c_void,
                        &mut alphas_ptr as *mut _ as *mut c_void,
                        &mut warms_ptr as *mut _ as *mut c_void,
                        &mut len_i as *mut _ as *mut c_void,
                        &mut first_i as *mut _ as *mut c_void,
                        &mut cur_i as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    self.stream.launch(&func, grid, block, 0, args)?;
                } else {
                    let mut high_ptr = d_high_opt
                        .as_ref()
                        .expect("device high buffer missing")
                        .as_device_ptr()
                        .as_raw();
                    let mut low_ptr = d_low_opt
                        .as_ref()
                        .expect("device low buffer missing")
                        .as_device_ptr()
                        .as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut high_ptr as *mut _ as *mut c_void,
                        &mut low_ptr as *mut _ as *mut c_void,
                        &mut periods_ptr as *mut _ as *mut c_void,
                        &mut alphas_ptr as *mut _ as *mut c_void,
                        &mut warms_ptr as *mut _ as *mut c_void,
                        &mut len_i as *mut _ as *mut c_void,
                        &mut first_i as *mut _ as *mut c_void,
                        &mut cur_i as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    self.stream.launch(&func, grid, block, 0, args)?;
                }
            }
            launched += cur;
        }

        self.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: len,
        })
    }

    pub fn cvi_batch_from_range_device_inplace(
        &self,
        d_range: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        d_periods: &DeviceBuffer<i32>,
        d_alphas: &DeviceBuffer<f32>,
        d_warm_indices: &DeviceBuffer<i32>,
        rows: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaCviError> {
        if len == 0 || rows == 0 {
            return Err(CudaCviError::InvalidInput("empty input".into()));
        }
        if first_valid >= len {
            return Err(CudaCviError::InvalidInput(
                "first_valid out of range".into(),
            ));
        }
        if d_range.len() < len {
            return Err(CudaCviError::InvalidInput("range buffer too small".into()));
        }
        if d_periods.len() < rows || d_alphas.len() < rows || d_warm_indices.len() < rows {
            return Err(CudaCviError::InvalidInput("params buffer too small".into()));
        }
        let needed = rows
            .checked_mul(len)
            .ok_or_else(|| CudaCviError::InvalidInput("size overflow".into()))?;
        if d_out.len() < needed {
            return Err(CudaCviError::InvalidInput("output buffer too small".into()));
        }

        let func = self
            .module
            .get_function("cvi_batch_from_range_f32")
            .map_err(|_| CudaCviError::MissingKernelSymbol {
                name: "cvi_batch_from_range_f32",
            })?;

        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            BatchKernelPolicy::Auto => 256,
        };
        let chunk = {
            let max_blocks: usize = 16_384;
            (rows).min(max_blocks * (block_x as usize))
        };

        let mut launched = 0usize;
        while launched < rows {
            let cur = (rows - launched).min(chunk);
            let (gx, grid): (u32, GridSize) = Self::grid_1d_for_elems(cur, block_x);
            let block: BlockSize = (block_x, 1, 1).into();
            self.validate_launch(gx, 1, 1, block_x, 1, 1)?;
            unsafe {
                let mut len_i = len as i32;
                let mut first_i = first_valid as i32;
                let mut cur_i = cur as i32;
                let mut range_ptr = d_range.as_device_ptr().as_raw();
                let mut periods_ptr = d_periods
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * std::mem::size_of::<i32>()) as u64);
                let mut alphas_ptr = d_alphas
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * std::mem::size_of::<f32>()) as u64);
                let mut warms_ptr = d_warm_indices
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * std::mem::size_of::<i32>()) as u64);
                let mut out_ptr = d_out
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * len * std::mem::size_of::<f32>()) as u64);
                let args: &mut [*mut c_void] = &mut [
                    &mut range_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut alphas_ptr as *mut _ as *mut c_void,
                    &mut warms_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut cur_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, args)?;
            }
            launched += cur;
        }
        Ok(())
    }

    fn first_valids_time_major(
        high_tm: &[f32],
        low_tm: &[f32],
        cols: usize,
        rows: usize,
    ) -> Result<Vec<i32>, CudaCviError> {
        let n = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaCviError::InvalidInput("rows*cols overflow".into()))?;
        if high_tm.len() != n || low_tm.len() != n {
            return Err(CudaCviError::InvalidInput(
                "time-major input length mismatch".into(),
            ));
        }
        let mut out = vec![-1i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                let h = high_tm[idx];
                let l = low_tm[idx];
                if !h.is_nan() && !l.is_nan() {
                    out[s] = t as i32;
                    break;
                }
            }
        }
        Ok(out)
    }

    pub fn cvi_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaCviError> {
        if period == 0 {
            return Err(CudaCviError::InvalidInput("period must be > 0".into()));
        }
        let first_valids = Self::first_valids_time_major(high_tm, low_tm, cols, rows)?;
        let warm = first_valids
            .iter()
            .copied()
            .filter(|&fv| fv >= 0)
            .map(|fv| fv as usize + (2 * period - 1))
            .max()
            .unwrap_or(0);
        if warm >= rows {
            return Err(CudaCviError::InvalidInput(
                "not enough rows for period/warmup".into(),
            ));
        }

        let total = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaCviError::InvalidInput("rows*cols overflow".into()))?;

        let f32b = std::mem::size_of::<f32>();
        let i32b = std::mem::size_of::<i32>();
        let need = total
            .checked_mul(3 * f32b)
            .and_then(|v| v.checked_add(cols.checked_mul(i32b)?))
            .ok_or_else(|| CudaCviError::InvalidInput("size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(need, headroom) {
            let (free, _) = mem_get_info().unwrap_or((0, 0));
            return Err(CudaCviError::OutOfMemory {
                required: need,
                free,
                headroom,
            });
        }

        let d_high = DeviceBuffer::from_slice(high_tm)?;
        let d_low = DeviceBuffer::from_slice(low_tm)?;
        let d_fv = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }?;

        let func = self
            .module
            .get_function("cvi_many_series_one_param_f32")
            .map_err(|_| CudaCviError::MissingKernelSymbol {
                name: "cvi_many_series_one_param_f32",
            })?;

        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            ManySeriesKernelPolicy::Auto => 256,
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let gx = grid_x.max(1);
        let grid: GridSize = (gx, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        self.validate_launch(gx, 1, 1, block_x, 1, 1)?;

        unsafe {
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
            let mut fv_ptr = d_fv.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut alpha_f = 2.0f32 / (period as f32 + 1.0f32);
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut alpha_f as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        self.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }
}

#[cfg(not(test))]
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const DEV_BATCH_LEN: usize = 65_536;
    const DEV_BATCH_PERIOD_SWEEP: (usize, usize, usize) = (2, 8192, 2);

    fn synth_hl_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if v.is_nan() {
                continue;
            }
            let x = i as f32 * 0.002f32;
            let off = (0.004 * x.sin()).abs() + 0.12;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    struct CviBatchDevState {
        cuda: CudaCvi,
        d_range: DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        d_periods: DeviceBuffer<i32>,
        d_alphas: DeviceBuffer<f32>,
        d_warms: DeviceBuffer<i32>,
        rows: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for CviBatchDevState {
        fn launch(&mut self) {
            self.cuda
                .cvi_batch_from_range_device_inplace(
                    &self.d_range,
                    self.len,
                    self.first_valid,
                    &self.d_periods,
                    &self.d_alphas,
                    &self.d_warms,
                    self.rows,
                    &mut self.d_out,
                )
                .expect("cvi batch dev kernel");
            self.cuda.stream().synchronize().expect("cvi sync");
        }
    }

    struct CviManyState {
        cuda: CudaCvi,
        d_high_tm: DeviceBuffer<f32>,
        d_low_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        alpha: f32,
        block_x: u32,
        grid_x: u32,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for CviManyState {
        fn launch(&mut self) {
            let func = self
                .cuda
                .module
                .get_function("cvi_many_series_one_param_f32")
                .expect("cvi_many_series_one_param_f32");
            let grid: GridSize = (self.grid_x.max(1), 1, 1).into();
            let block: BlockSize = (self.block_x, 1, 1).into();
            unsafe {
                let mut high_ptr = self.d_high_tm.as_device_ptr().as_raw();
                let mut low_ptr = self.d_low_tm.as_device_ptr().as_raw();
                let mut fv_ptr = self.d_first_valids.as_device_ptr().as_raw();
                let mut period_i = self.period as i32;
                let mut alpha_f = self.alpha;
                let mut num_series_i = self.cols as i32;
                let mut series_len_i = self.rows as i32;
                let mut out_ptr = self.d_out_tm.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut high_ptr as *mut _ as *mut c_void,
                    &mut low_ptr as *mut _ as *mut c_void,
                    &mut fv_ptr as *mut _ as *mut c_void,
                    &mut period_i as *mut _ as *mut c_void,
                    &mut alpha_f as *mut _ as *mut c_void,
                    &mut num_series_i as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.cuda
                    .stream
                    .launch(&func, grid, block, 0, args)
                    .expect("cvi many-series launch");
            }
            self.cuda.stream.synchronize().expect("cvi sync");
        }
    }

    fn prep_one_series_many_params_dev() -> Box<dyn CudaBenchState> {
        let cuda = CudaCvi::new(0).unwrap();
        let len = DEV_BATCH_LEN;
        let close = gen_series(len);
        let (high, low) = synth_hl_from_close(&close);
        let first_valid = CudaCvi::first_valid_hl(&high, &low).expect("cvi first_valid");
        let sweep = CviBatchRange {
            period: DEV_BATCH_PERIOD_SWEEP,
        };
        let (ps, pe, pst) = sweep.period;
        let periods: Vec<usize> = if pst == 0 || ps == pe {
            vec![ps]
        } else if ps < pe {
            (ps..=pe).step_by(pst).collect()
        } else {
            Vec::new()
        };
        let rows = periods.len();
        let periods_host: Vec<i32> = periods.iter().map(|&p| p as i32).collect();
        let alphas_host: Vec<f32> = periods
            .iter()
            .map(|&p| 2.0f32 / (p as f32 + 1.0f32))
            .collect();
        let warms_host: Vec<i32> = periods
            .iter()
            .map(|&p| (first_valid + (2 * p - 1)) as i32)
            .collect();

        let mut range = vec![0f32; len];
        for i in 0..len {
            range[i] = high[i] - low[i];
        }
        let elems_out = rows.checked_mul(len).expect("cvi bench size overflow");
        let d_range = unsafe { DeviceBuffer::from_slice_async(&range, cuda.stream()) }.unwrap();
        let d_periods =
            unsafe { DeviceBuffer::from_slice_async(&periods_host, cuda.stream()) }.unwrap();
        let d_alphas =
            unsafe { DeviceBuffer::from_slice_async(&alphas_host, cuda.stream()) }.unwrap();
        let d_warms =
            unsafe { DeviceBuffer::from_slice_async(&warms_host, cuda.stream()) }.unwrap();
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems_out, cuda.stream()) }.unwrap();
        cuda.stream().synchronize().unwrap();

        Box::new(CviBatchDevState {
            cuda,
            d_range,
            len,
            first_valid,
            d_periods,
            d_alphas,
            d_warms,
            rows,
            d_out,
        })
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let (cols, rows, period) = (256usize, 262_144usize, 14usize);
        let mut base = vec![f32::NAN; cols * rows];
        for s in 0..cols {
            for t in s..rows {
                let x = (t as f32) + (s as f32) * 0.2;
                base[t * cols + s] = (x * 0.0017).sin() + 0.00015 * x;
            }
        }
        let mut high_tm = base.clone();
        let mut low_tm = base.clone();
        for s in 0..cols {
            for t in 0..rows {
                let v = base[t * cols + s];
                if v.is_nan() {
                    continue;
                }
                let x = (t as f32) * 0.002;
                let off = (0.004 * x.cos()).abs() + 0.11;
                high_tm[t * cols + s] = v + off;
                low_tm[t * cols + s] = v - off;
            }
        }
        let first_valids = CudaCvi::first_valids_time_major(&high_tm, &low_tm, cols, rows)
            .expect("cvi first_valids");
        let cuda = CudaCvi::new(0).unwrap();
        let d_high_tm = DeviceBuffer::from_slice(&high_tm).expect("d_high_tm");
        let d_low_tm = DeviceBuffer::from_slice(&low_tm).expect("d_low_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let elems = cols * rows;
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_out_tm");
        let alpha = 2.0f32 / (period as f32 + 1.0f32);
        let block_x: u32 = match cuda.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            ManySeriesKernelPolicy::Auto => 256,
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        cuda.stream.synchronize().expect("sync after prep");
        Box::new(CviManyState {
            cuda,
            d_high_tm,
            d_low_tm,
            d_first_valids,
            cols,
            rows,
            period,
            alpha,
            block_x,
            grid_x,
            d_out_tm,
        })
    }

    fn bytes_one_series_many_params(len: usize, rows: usize) -> usize {
        let in_bytes = len * std::mem::size_of::<f32>();
        let param_bytes = rows * (std::mem::size_of::<i32>() * 2 + std::mem::size_of::<f32>());
        let out_bytes = rows * len * std::mem::size_of::<f32>();
        in_bytes + param_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let (cols, rows) = (256usize, 262_144usize);
        (2 * cols * rows + cols + cols * rows) * std::mem::size_of::<f32>() + 64 * 1024 * 1024
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let rows_dev = (DEV_BATCH_PERIOD_SWEEP.0..=DEV_BATCH_PERIOD_SWEEP.1)
            .step_by(DEV_BATCH_PERIOD_SWEEP.2.max(1))
            .count();
        let scen_batch_dev = CudaBenchScenario::new(
            "cvi",
            "one_series_many_params",
            "cvi_cuda_batch_dev_inplace",
            "64k_x_4096",
            prep_one_series_many_params_dev,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params(DEV_BATCH_LEN, rows_dev));
        let scen_many = CudaBenchScenario::new(
            "cvi",
            "many_series_one_param",
            "cvi_cuda_many_series_one_param_dev",
            "256x262k",
            prep_many_series_one_param,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_many_series_one_param());
        vec![scen_batch_dev, scen_many]
    }
}
