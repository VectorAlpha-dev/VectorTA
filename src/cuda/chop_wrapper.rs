#![cfg(feature = "cuda")]

use crate::indicators::chop::{ChopBatchRange, ChopParams};
use crate::indicators::willr::build_willr_gpu_tables;
use cust::context::{CacheConfig, Context};
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, DeviceCopy, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::sync::Arc;
use thiserror::Error;

const CHOP_REG_RING_MAX: usize = 64;

const PINNED_STAGING_THRESHOLD: usize = 1 << 20;

#[derive(Debug, Error)]
pub enum CudaChopError {
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

pub struct DeviceArrayF32 {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}
impl DeviceArrayF32 {
    #[inline]
    pub fn device_ptr(&self) -> u64 {
        self.buf.as_device_ptr().as_raw() as u64
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaChop {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaChop {
    pub fn new(device_id: usize) -> Result<Self, CudaChopError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/chop_kernel.ptx"));

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
    fn upload_slice_async<T: DeviceCopy>(
        &self,
        slice: &[T],
    ) -> Result<DeviceBuffer<T>, CudaChopError> {
        use std::mem::size_of;
        let bytes = slice
            .len()
            .checked_mul(size_of::<T>())
            .ok_or_else(|| CudaChopError::InvalidInput("byte size overflow".into()))?;
        if bytes >= PINNED_STAGING_THRESHOLD {
            let mut pinned: LockedBuffer<T> = unsafe { LockedBuffer::uninitialized(slice.len()) }?;
            pinned.as_mut_slice().copy_from_slice(slice);

            let mut d = unsafe { DeviceBuffer::uninitialized_async(slice.len(), &self.stream) }?;
            unsafe {
                d.async_copy_from(pinned.as_slice(), &self.stream)?;
            }
            Ok(d)
        } else {
            unsafe { DeviceBuffer::from_slice_async(slice, &self.stream) }.map_err(Into::into)
        }
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaChopError> {
        match mem_get_info() {
            Ok((free, _total)) => {
                if required_bytes.saturating_add(headroom_bytes) <= free {
                    Ok(())
                } else {
                    Err(CudaChopError::OutOfMemory {
                        required: required_bytes,
                        free: free,
                        headroom: headroom_bytes,
                    })
                }
            }
            Err(_) => Ok(()),
        }
    }

    fn expand_grid(range: &ChopBatchRange) -> Result<Vec<ChopParams>, CudaChopError> {
        let (ps, pe, pt) = range.period;
        let (ss, se, st) = range.scalar;
        let (ds, de, dt) = range.drift;

        let periods: Vec<usize> = if pt == 0 || ps == pe {
            vec![ps]
        } else if ps < pe {
            (ps..=pe).step_by(pt).collect()
        } else {
            let mut v = Vec::new();
            let mut x = ps;
            while x >= pe {
                v.push(x);
                if x < pe + pt {
                    break;
                }
                x -= pt;
                if x == 0 {
                    break;
                }
            }
            v
        };

        let scalars: Vec<f64> = if st.abs() < 1e-12 || (ss - se).abs() < f64::EPSILON {
            vec![ss]
        } else if ss <= se && st > 0.0 {
            let mut v = Vec::new();
            let mut x = ss;
            while x <= se + 1e-12 {
                v.push(x);
                x += st;
            }
            v
        } else if ss >= se && st < 0.0 {
            let mut v = Vec::new();
            let mut x = ss;
            while x >= se - 1e-12 {
                v.push(x);
                x += st;
            }
            v
        } else {
            return Err(CudaChopError::InvalidInput(
                "scalar range step direction invalid".into(),
            ));
        };
        let drifts: Vec<usize> = if dt == 0 || ds == de {
            vec![ds]
        } else if ds < de {
            (ds..=de).step_by(dt).collect()
        } else {
            let mut v = Vec::new();
            let mut x = ds;
            while x >= de {
                v.push(x);
                if x < de + dt {
                    break;
                }
                x -= dt;
                if x == 0 {
                    break;
                }
            }
            v
        };
        let cap = periods
            .len()
            .checked_mul(scalars.len())
            .and_then(|x| x.checked_mul(drifts.len()))
            .ok_or_else(|| CudaChopError::InvalidInput("rows*cols overflow".into()))?;
        let mut combos = Vec::with_capacity(cap);
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
        if combos.is_empty() {
            return Err(CudaChopError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        Ok(combos)
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

        let combos = Self::expand_grid(sweep)?;

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

        let tables = build_willr_gpu_tables(high_f32, low_f32);

        let periods_i32: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let drifts_i32: Vec<i32> = combos.iter().map(|c| c.drift.unwrap() as i32).collect();
        let scalars_f32: Vec<f32> = combos.iter().map(|c| c.scalar.unwrap() as f32).collect();

        let out_elems = combos
            .len()
            .checked_mul(n)
            .ok_or_else(|| CudaChopError::InvalidInput("rows*cols overflow".into()))?;

        let mut bytes: usize = 0;
        bytes = bytes
            .checked_add(
                high_f32
                    .len()
                    .checked_add(low_f32.len())
                    .and_then(|x| x.checked_add(close_f32.len()))
                    .and_then(|x| x.checked_mul(4))
                    .ok_or_else(|| CudaChopError::InvalidInput("byte size overflow".into()))?,
            )
            .ok_or_else(|| CudaChopError::InvalidInput("byte size overflow".into()))?;
        bytes = bytes
            .checked_add(
                periods_i32
                    .len()
                    .checked_add(drifts_i32.len())
                    .and_then(|x| x.checked_mul(4))
                    .ok_or_else(|| CudaChopError::InvalidInput("byte size overflow".into()))?,
            )
            .ok_or_else(|| CudaChopError::InvalidInput("byte size overflow".into()))?;
        bytes = bytes
            .checked_add(
                scalars_f32
                    .len()
                    .checked_mul(4)
                    .ok_or_else(|| CudaChopError::InvalidInput("byte size overflow".into()))?,
            )
            .ok_or_else(|| CudaChopError::InvalidInput("byte size overflow".into()))?;
        bytes = bytes
            .checked_add(
                tables
                    .log2
                    .len()
                    .checked_mul(4)
                    .ok_or_else(|| CudaChopError::InvalidInput("byte size overflow".into()))?,
            )
            .ok_or_else(|| CudaChopError::InvalidInput("byte size overflow".into()))?;
        bytes = bytes
            .checked_add(
                tables
                    .level_offsets
                    .len()
                    .checked_mul(4)
                    .ok_or_else(|| CudaChopError::InvalidInput("byte size overflow".into()))?,
            )
            .ok_or_else(|| CudaChopError::InvalidInput("byte size overflow".into()))?;
        bytes = bytes
            .checked_add(
                tables
                    .st_max
                    .len()
                    .checked_mul(4)
                    .ok_or_else(|| CudaChopError::InvalidInput("byte size overflow".into()))?,
            )
            .ok_or_else(|| CudaChopError::InvalidInput("byte size overflow".into()))?;
        bytes = bytes
            .checked_add(
                tables
                    .st_min
                    .len()
                    .checked_mul(4)
                    .ok_or_else(|| CudaChopError::InvalidInput("byte size overflow".into()))?,
            )
            .ok_or_else(|| CudaChopError::InvalidInput("byte size overflow".into()))?;
        bytes = bytes
            .checked_add(
                tables
                    .nan_psum
                    .len()
                    .checked_mul(4)
                    .ok_or_else(|| CudaChopError::InvalidInput("byte size overflow".into()))?,
            )
            .ok_or_else(|| CudaChopError::InvalidInput("byte size overflow".into()))?;
        bytes = bytes
            .checked_add(
                out_elems
                    .checked_mul(4)
                    .ok_or_else(|| CudaChopError::InvalidInput("byte size overflow".into()))?,
            )
            .ok_or_else(|| CudaChopError::InvalidInput("byte size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        Self::will_fit(bytes, headroom)?;

        let d_high = self.upload_slice_async(high_f32)?;
        let d_low = self.upload_slice_async(low_f32)?;
        let d_close = self.upload_slice_async(close_f32)?;
        let d_periods = self.upload_slice_async(&periods_i32)?;
        let d_drifts = self.upload_slice_async(&drifts_i32)?;
        let d_scalars = self.upload_slice_async(&scalars_f32)?;

        let d_log2 = self.upload_slice_async(&tables.log2)?;
        let d_offsets = self.upload_slice_async(&tables.level_offsets)?;
        let d_st_max = self.upload_slice_async(&tables.st_max)?;
        let d_st_min = self.upload_slice_async(&tables.st_min)?;
        let d_nan_psum = self.upload_slice_async(&tables.nan_psum)?;

        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_elems, &self.stream) }?;

        let mut func = self.module.get_function("chop_batch_f32").map_err(|_| {
            CudaChopError::MissingKernelSymbol {
                name: "chop_batch_f32",
            }
        })?;

        let shared_bytes: usize = if max_period <= CHOP_REG_RING_MAX {
            0
        } else {
            max_period
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| CudaChopError::InvalidInput("shared memory size overflow".into()))?
        };
        let _ = if shared_bytes == 0 {
            func.set_cache_config(CacheConfig::PreferL1)
        } else {
            func.set_cache_config(CacheConfig::PreferShared)
        };

        let rows = combos.len();
        let mut launched = 0usize;
        while launched < rows {
            let n_this = (rows - launched).min(65_535);
            let grid: GridSize = (n_this as u32, 1u32, 1u32).into();
            let block: BlockSize = (256u32, 1u32, 1u32).into();
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
                )?;
            }
            launched += n_this;
        }

        self.stream.synchronize()?;

        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows,
                cols: n,
                ctx: self.context.clone(),
                device_id: self.device_id,
            },
            combos,
        ))
    }

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
        let n = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaChopError::InvalidInput("rows*cols overflow".into()))?;
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

        for s in 0..cols {
            let fv = first_valids[s];
            if fv < 0 {
                return Err(CudaChopError::InvalidInput("all values are NaN".into()));
            }
            if rows - (fv as usize) < period {
                return Err(CudaChopError::InvalidInput("not enough valid data".into()));
            }
        }

        let atr_rows = rows
            .checked_add(1)
            .ok_or_else(|| CudaChopError::InvalidInput("rows overflow".into()))?;
        let atr_len = atr_rows
            .checked_mul(cols)
            .ok_or_else(|| CudaChopError::InvalidInput("rows*cols overflow".into()))?;
        let mut atr_psum_tm = vec![0f32; atr_len];
        {
            let inv_drift = 1.0f64 / (drift as f64);
            for s in 0..cols {
                let fv = first_valids[s] as usize;
                let mut prev_close = close_tm_f32[fv * cols + s] as f64;
                let mut rma_atr: f64 = f64::NAN;
                let mut sum_tr = 0.0f64;
                let mut acc = 0.0f64;

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

        let mut bytes: usize = 0;
        bytes = bytes
            .checked_add(
                high_tm_f32
                    .len()
                    .checked_add(low_tm_f32.len())
                    .and_then(|x| x.checked_mul(4))
                    .ok_or_else(|| CudaChopError::InvalidInput("byte size overflow".into()))?,
            )
            .ok_or_else(|| CudaChopError::InvalidInput("byte size overflow".into()))?;
        bytes = bytes
            .checked_add(
                atr_psum_tm
                    .len()
                    .checked_mul(4)
                    .ok_or_else(|| CudaChopError::InvalidInput("byte size overflow".into()))?,
            )
            .ok_or_else(|| CudaChopError::InvalidInput("byte size overflow".into()))?;
        bytes = bytes
            .checked_add(
                first_valids
                    .len()
                    .checked_mul(4)
                    .ok_or_else(|| CudaChopError::InvalidInput("byte size overflow".into()))?,
            )
            .ok_or_else(|| CudaChopError::InvalidInput("byte size overflow".into()))?;
        bytes = bytes
            .checked_add(
                n.checked_mul(4)
                    .ok_or_else(|| CudaChopError::InvalidInput("byte size overflow".into()))?,
            )
            .ok_or_else(|| CudaChopError::InvalidInput("byte size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        Self::will_fit(bytes, headroom)?;

        let d_high = self.upload_slice_async(high_tm_f32)?;
        let d_low = self.upload_slice_async(low_tm_f32)?;
        let d_psum = self.upload_slice_async(&atr_psum_tm)?;
        let d_first = self.upload_slice_async(&first_valids)?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(n, &self.stream) }?;

        let mut func = self
            .module
            .get_function("chop_many_series_one_param_f32")
            .map_err(|_| CudaChopError::MissingKernelSymbol {
                name: "chop_many_series_one_param_f32",
            })?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);

        let block: BlockSize = (256u32, 1u32, 1u32).into();
        let grid: GridSize = (((cols as u32 + 255) / 256).max(1), 1u32, 1u32).into();
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
            )?;
        }

        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
            ctx: self.context.clone(),
            device_id: self.device_id,
        })
    }

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
        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(arr.len()) }?;
        unsafe {
            arr.buf.async_copy_to(pinned.as_mut_slice(), &self.stream)?;
        }
        self.stream.synchronize()?;
        self.stream.synchronize()?;
        out.copy_from_slice(pinned.as_slice());
        Ok((arr.rows, arr.cols, combos))
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use std::ffi::c_void;

    const ONE_SERIES_LEN: usize = 50_000;
    const PARAM_SWEEP: usize = 128;
    const MANY_COLS: usize = 128;
    const MANY_ROWS: usize = 8192;

    fn synth_hl_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if !v.is_finite() {
                continue;
            }
            let x = i as f32 * 0.0021;
            let off = (0.0033 * x.sin()).abs() + 0.12;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    fn bytes_sparse_tables(n: usize) -> usize {
        let levels = (usize::BITS as usize).saturating_sub(n.max(1).leading_zeros() as usize);
        let st_elems = n.saturating_mul(levels);

        let st_bytes = 2usize
            .saturating_mul(st_elems)
            .saturating_mul(std::mem::size_of::<f32>());

        let meta_i32 = (n + 1).saturating_mul(2).saturating_add(levels + 1);
        let meta_bytes = meta_i32.saturating_mul(std::mem::size_of::<i32>());
        st_bytes + meta_bytes
    }

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = 3 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = PARAM_SWEEP * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let params_bytes = PARAM_SWEEP
            * (std::mem::size_of::<i32>()
                + std::mem::size_of::<i32>()
                + std::mem::size_of::<f32>());
        in_bytes + params_bytes + bytes_sparse_tables(ONE_SERIES_LEN) + out_bytes + 64 * 1024 * 1024
    }

    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_COLS * MANY_ROWS;
        let in_bytes = 2 * elems * std::mem::size_of::<f32>();
        let psum_bytes = (MANY_ROWS + 1) * MANY_COLS * std::mem::size_of::<f32>();
        let first_bytes = MANY_COLS * std::mem::size_of::<i32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        in_bytes + psum_bytes + first_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct ChopBatchDeviceState {
        cuda: CudaChop,
        func: Function<'static>,
        d_high: DeviceBuffer<f32>,
        d_low: DeviceBuffer<f32>,
        d_close: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_drifts: DeviceBuffer<i32>,
        d_scalars: DeviceBuffer<f32>,
        d_log2: DeviceBuffer<i32>,
        d_offsets: DeviceBuffer<i32>,
        d_st_max: DeviceBuffer<f32>,
        d_st_min: DeviceBuffer<f32>,
        d_nan_psum: DeviceBuffer<i32>,
        d_out: DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        levels: i32,
        max_period: i32,
        rows: usize,
        shared_bytes: u32,
        block_x: u32,
    }
    impl CudaBenchState for ChopBatchDeviceState {
        fn launch(&mut self) {
            let combos_per_launch = 65_535usize;
            let mut row0 = 0usize;
            while row0 < self.rows {
                let n = (self.rows - row0).min(combos_per_launch);
                let grid: GridSize = (n as u32, 1, 1).into();
                let block: BlockSize = (self.block_x, 1, 1).into();
                unsafe {
                    let mut high_ptr = self.d_high.as_device_ptr().as_raw();
                    let mut low_ptr = self.d_low.as_device_ptr().as_raw();
                    let mut close_ptr = self.d_close.as_device_ptr().as_raw();
                    let mut periods_ptr = self
                        .d_periods
                        .as_device_ptr()
                        .offset(row0 as isize)
                        .as_raw();
                    let mut drifts_ptr =
                        self.d_drifts.as_device_ptr().offset(row0 as isize).as_raw();
                    let mut scalars_ptr = self
                        .d_scalars
                        .as_device_ptr()
                        .offset(row0 as isize)
                        .as_raw();
                    let mut log2_ptr = self.d_log2.as_device_ptr().as_raw();
                    let mut offs_ptr = self.d_offsets.as_device_ptr().as_raw();
                    let mut stmax_ptr = self.d_st_max.as_device_ptr().as_raw();
                    let mut stmin_ptr = self.d_st_min.as_device_ptr().as_raw();
                    let mut npsum_ptr = self.d_nan_psum.as_device_ptr().as_raw();
                    let mut len_i = self.len as i32;
                    let mut first_i = self.first_valid as i32;
                    let mut levels_i = self.levels;
                    let mut n_i = n as i32;
                    let mut maxp_i = self.max_period;
                    let mut out_ptr = self
                        .d_out
                        .as_device_ptr()
                        .offset((row0 * self.len) as isize)
                        .as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut high_ptr as *mut _ as *mut c_void,
                        &mut low_ptr as *mut _ as *mut c_void,
                        &mut close_ptr as *mut _ as *mut c_void,
                        &mut periods_ptr as *mut _ as *mut c_void,
                        &mut drifts_ptr as *mut _ as *mut c_void,
                        &mut scalars_ptr as *mut _ as *mut c_void,
                        &mut log2_ptr as *mut _ as *mut c_void,
                        &mut offs_ptr as *mut _ as *mut c_void,
                        &mut stmax_ptr as *mut _ as *mut c_void,
                        &mut stmin_ptr as *mut _ as *mut c_void,
                        &mut npsum_ptr as *mut _ as *mut c_void,
                        &mut len_i as *mut _ as *mut c_void,
                        &mut first_i as *mut _ as *mut c_void,
                        &mut levels_i as *mut _ as *mut c_void,
                        &mut n_i as *mut _ as *mut c_void,
                        &mut maxp_i as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    self.cuda
                        .stream
                        .launch(&self.func, grid, block, self.shared_bytes, args)
                        .expect("chop_batch launch");
                }
                row0 += n;
            }
            self.cuda.stream.synchronize().expect("chop_batch sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaChop::new(0).expect("cuda chop");
        let close = gen_series(ONE_SERIES_LEN);
        let (high, low) = synth_hl_from_close(&close);
        let sweep = ChopBatchRange {
            period: (5, 5 + PARAM_SWEEP - 1, 1),
            scalar: (100.0, 100.0, 0.0),
            drift: (1, 1, 0),
        };
        let combos = CudaChop::expand_grid(&sweep).expect("expand_grid");
        let rows = combos.len();
        let first_valid = (0..ONE_SERIES_LEN)
            .find(|&i| high[i].is_finite() && low[i].is_finite() && close[i].is_finite())
            .unwrap_or(0);
        let max_period = combos
            .iter()
            .map(|c| c.period.unwrap_or(0) as i32)
            .max()
            .unwrap_or(0);

        let periods_i32: Vec<i32> = combos
            .iter()
            .map(|c| c.period.unwrap_or(14) as i32)
            .collect();
        let drifts_i32: Vec<i32> = combos.iter().map(|c| c.drift.unwrap_or(1) as i32).collect();
        let scalars_f32: Vec<f32> = combos
            .iter()
            .map(|c| c.scalar.unwrap_or(100.0) as f32)
            .collect();

        let tables = build_willr_gpu_tables(&high, &low);
        let levels = (tables.level_offsets.len() - 1) as i32;
        let shared_bytes: u32 = if (max_period as usize) <= CHOP_REG_RING_MAX {
            0
        } else {
            (max_period as u32) * (std::mem::size_of::<f32>() as u32)
        };

        let d_high: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::from_slice_async(&high, &cuda.stream) }.expect("d_high");
        let d_low: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::from_slice_async(&low, &cuda.stream) }.expect("d_low");
        let d_close: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::from_slice_async(&close, &cuda.stream) }.expect("d_close");

        let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("d_periods");
        let d_drifts = DeviceBuffer::from_slice(&drifts_i32).expect("d_drifts");
        let d_scalars = DeviceBuffer::from_slice(&scalars_f32).expect("d_scalars");

        let d_log2 = DeviceBuffer::from_slice(&tables.log2).expect("d_log2");
        let d_offsets = DeviceBuffer::from_slice(&tables.level_offsets).expect("d_offsets");
        let d_st_max = DeviceBuffer::from_slice(&tables.st_max).expect("d_st_max");
        let d_st_min = DeviceBuffer::from_slice(&tables.st_min).expect("d_st_min");
        let d_nan_psum = DeviceBuffer::from_slice(&tables.nan_psum).expect("d_nan_psum");

        let func = cuda
            .module
            .get_function("chop_batch_f32")
            .expect("chop_batch_f32");
        let mut func: Function<'static> = unsafe { std::mem::transmute(func) };
        if shared_bytes == 0 {
            let _ = func.set_cache_config(CacheConfig::PreferL1);
        } else {
            let _ = func.set_cache_config(CacheConfig::PreferShared);
        }

        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(rows * ONE_SERIES_LEN, &cuda.stream) }
                .expect("d_out");

        cuda.stream.synchronize().expect("sync after prep");
        Box::new(ChopBatchDeviceState {
            cuda,
            func,
            d_high,
            d_low,
            d_close,
            d_periods,
            d_drifts,
            d_scalars,
            d_log2,
            d_offsets,
            d_st_max,
            d_st_min,
            d_nan_psum,
            d_out,
            len: ONE_SERIES_LEN,
            first_valid,
            levels,
            max_period,
            rows,
            shared_bytes,
            block_x: 256,
        })
    }

    struct ChopManySeriesDeviceState {
        cuda: CudaChop,
        func: Function<'static>,
        d_high: DeviceBuffer<f32>,
        d_low: DeviceBuffer<f32>,
        d_psum: DeviceBuffer<f32>,
        d_first: DeviceBuffer<i32>,
        d_out: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        period: i32,
        scalar: f32,
        block_x: u32,
    }
    impl CudaBenchState for ChopManySeriesDeviceState {
        fn launch(&mut self) {
            let grid: GridSize = (((self.cols as u32 + 255) / 256).max(1), 1, 1).into();
            let block: BlockSize = (self.block_x, 1, 1).into();
            unsafe {
                let mut high_ptr = self.d_high.as_device_ptr().as_raw();
                let mut low_ptr = self.d_low.as_device_ptr().as_raw();
                let mut psum_ptr = self.d_psum.as_device_ptr().as_raw();
                let mut first_ptr = self.d_first.as_device_ptr().as_raw();
                let mut cols_i = self.cols as i32;
                let mut rows_i = self.rows as i32;
                let mut period_i = self.period;
                let mut scalar_f = self.scalar;
                let mut out_ptr = self.d_out.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut high_ptr as *mut _ as *mut c_void,
                    &mut low_ptr as *mut _ as *mut c_void,
                    &mut psum_ptr as *mut _ as *mut c_void,
                    &mut first_ptr as *mut _ as *mut c_void,
                    &mut cols_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut period_i as *mut _ as *mut c_void,
                    &mut scalar_f as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.cuda
                    .stream
                    .launch(&self.func, grid, block, 0, args)
                    .expect("chop_many_series launch");
            }
            self.cuda
                .stream
                .synchronize()
                .expect("chop_many_series sync");
        }
    }

    fn build_atr_psum_tm(
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        first_valids: &[i32],
        drift: usize,
    ) -> Vec<f32> {
        let mut out = vec![0f32; (rows + 1) * cols];
        let inv_drift = 1.0f64 / (drift as f64);
        for s in 0..cols {
            let fv = first_valids[s] as usize;
            let mut prev_close = close_tm[fv * cols + s] as f64;
            let mut rma_atr: f64 = f64::NAN;
            let mut sum_tr = 0.0f64;
            let mut acc = 0.0f64;
            for r in fv..rows {
                let hi = high_tm[r * cols + s] as f64;
                let lo = low_tm[r * cols + s] as f64;
                let cl = close_tm[r * cols + s] as f64;
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
                if current_atr.is_finite() {
                    acc += current_atr;
                }
                out[(r + 1) * cols + s] = acc as f32;
            }
        }
        out
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaChop::new(0).expect("cuda chop");
        let close_tm = gen_time_major_prices(MANY_COLS, MANY_ROWS);
        let mut high_tm = close_tm.clone();
        let mut low_tm = close_tm.clone();
        for r in 0..MANY_ROWS {
            for s in 0..MANY_COLS {
                let idx = r * MANY_COLS + s;
                let v = close_tm[idx];
                if !v.is_finite() {
                    continue;
                }
                let x = r as f32 * 0.0019 + s as f32 * 0.07;
                let off = (0.0027 * x.sin()).abs() + 0.08;
                high_tm[idx] = v + off;
                low_tm[idx] = v - off;
            }
        }
        let first_valids: Vec<i32> = (0..MANY_COLS).map(|s| s as i32).collect();
        let drift = 1usize;
        let atr_psum_tm = build_atr_psum_tm(
            &high_tm,
            &low_tm,
            &close_tm,
            MANY_COLS,
            MANY_ROWS,
            &first_valids,
            drift,
        );
        let period = 14i32;
        let scalar = 100.0f32;

        let d_high: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::from_slice_async(&high_tm, &cuda.stream) }.expect("d_high_tm");
        let d_low: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::from_slice_async(&low_tm, &cuda.stream) }.expect("d_low_tm");
        let d_psum: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::from_slice_async(&atr_psum_tm, &cuda.stream) }
                .expect("d_psum_tm");
        let d_first: DeviceBuffer<i32> = DeviceBuffer::from_slice(&first_valids).expect("d_first");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(MANY_COLS * MANY_ROWS, &cuda.stream) }
                .expect("d_out_tm");

        let func = cuda
            .module
            .get_function("chop_many_series_one_param_f32")
            .expect("chop_many_series_one_param_f32");
        let mut func: Function<'static> = unsafe { std::mem::transmute(func) };
        let _ = func.set_cache_config(CacheConfig::PreferL1);

        cuda.stream.synchronize().expect("sync after prep");
        Box::new(ChopManySeriesDeviceState {
            cuda,
            func,
            d_high,
            d_low,
            d_psum,
            d_first,
            d_out,
            cols: MANY_COLS,
            rows: MANY_ROWS,
            period,
            scalar,
            block_x: 256,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "chop",
                "one_series_many_params",
                "chop_cuda_batch_dev",
                "50k_x_128",
                prep_one_series_many_params,
            )
            .with_mem_required(bytes_one_series_many_params())
            .with_sample_size(10),
            CudaBenchScenario::new(
                "chop",
                "many_series_one_param",
                "chop_cuda_many_series_one_param",
                "128x8k",
                prep_many_series_one_param,
            )
            .with_mem_required(bytes_many_series_one_param())
            .with_sample_size(10),
        ]
    }
}
