#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::mab::{MabBatchRange, MabParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys as cu;
use std::env;
use std::ffi::c_void;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaMabError {
    #[error("CUDA error: {0}")]
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

pub struct DeviceArrayF32Triplet {
    pub upper: DeviceArrayF32,
    pub middle: DeviceArrayF32,
    pub lower: DeviceArrayF32,
}

impl DeviceArrayF32Triplet {
    #[inline]
    pub fn rows(&self) -> usize {
        self.upper.rows
    }
    #[inline]
    pub fn cols(&self) -> usize {
        self.upper.cols
    }
}

pub struct CudaMab {
    module: Module,
    stream: Stream,
    context: std::sync::Arc<Context>,
    device_id: u32,
}

impl CudaMab {
    pub fn new(device_id: usize) -> Result<Self, CudaMabError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = std::sync::Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/mab_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                {
                    m
                } else {
                    Module::from_ptx(ptx, &[])?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
        })
    }

    fn compute_ma_host(
        ma_type: &str,
        prices_f32: &[f32],
        period: usize,
    ) -> Result<Vec<f32>, CudaMabError> {
        use crate::indicators::moving_averages::ema::{ema, EmaInput, EmaParams};
        use crate::indicators::moving_averages::sma::{sma, SmaInput, SmaParams};
        let prices: Vec<f64> = prices_f32.iter().map(|&v| v as f64).collect();
        let n = prices.len();
        if period == 0 || period > n {
            return Err(CudaMabError::InvalidInput("invalid period".into()));
        }
        let out_f64 = match ma_type.to_ascii_lowercase().as_str() {
            "ema" => {
                ema(&EmaInput::from_slice(
                    &prices,
                    EmaParams {
                        period: Some(period),
                    },
                ))
                .map_err(|e| CudaMabError::InvalidInput(e.to_string()))?
                .values
            }
            _ => {
                sma(&SmaInput::from_slice(
                    &prices,
                    SmaParams {
                        period: Some(period),
                    },
                ))
                .map_err(|e| CudaMabError::InvalidInput(e.to_string()))?
                .values
            }
        };
        Ok(out_f64.into_iter().map(|v| v as f32).collect())
    }

    fn build_prefixes_single(prices: &[f32]) -> (Vec<f64>, Vec<i32>) {
        let len = prices.len();
        let mut pcs = vec![0.0f64; len + 1];
        let mut pnan = vec![0i32; len + 1];
        let mut acc_s = 0.0f64;
        let mut acc_nan = 0i32;
        for i in 0..len {
            let x = prices[i] as f64;
            if x.is_nan() {
                acc_nan += 1;
            } else {
                acc_s += x;
            }
            pcs[i + 1] = acc_s;
            pnan[i + 1] = acc_nan;
        }
        (pcs, pnan)
    }

    fn compute_ma_host_time_major(
        ma_type: &str,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<Vec<f32>, CudaMabError> {
        use crate::indicators::moving_averages::ema::{ema, EmaInput, EmaParams};
        use crate::indicators::moving_averages::sma::{sma, SmaInput, SmaParams};
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaMabError::InvalidInput("time-major dims overflow".into()))?;
        if data_tm_f32.len() != expected {
            return Err(CudaMabError::InvalidInput(
                "time-major dims mismatch".into(),
            ));
        }
        let mut out_tm = vec![f32::NAN; expected];
        for s in 0..cols {
            let mut col = vec![f64::NAN; rows];
            for r in 0..rows {
                col[r] = data_tm_f32[r * cols + s] as f64;
            }
            let vals = match ma_type.to_ascii_lowercase().as_str() {
                "ema" => {
                    ema(&EmaInput::from_slice(
                        &col,
                        EmaParams {
                            period: Some(period),
                        },
                    ))
                    .map_err(|e| CudaMabError::InvalidInput(e.to_string()))?
                    .values
                }
                _ => {
                    sma(&SmaInput::from_slice(
                        &col,
                        SmaParams {
                            period: Some(period),
                        },
                    ))
                    .map_err(|e| CudaMabError::InvalidInput(e.to_string()))?
                    .values
                }
            };
            for r in 0..rows {
                out_tm[r * cols + s] = vals[r] as f32;
            }
        }
        Ok(out_tm)
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Ok((free, _total)) = mem_get_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    pub fn mab_batch_device_sma(
        &self,
        d_pref_close_sum: &DeviceBuffer<f64>,
        d_pref_close_nan: &DeviceBuffer<i32>,
        d_fast_periods: &DeviceBuffer<i32>,
        d_slow_periods: &DeviceBuffer<i32>,
        d_devups: &DeviceBuffer<f32>,
        d_devdns: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        rows: usize,
        d_upper: &mut DeviceBuffer<f32>,
        d_middle: &mut DeviceBuffer<f32>,
        d_lower: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMabError> {
        let cur_dev = unsafe {
            let mut dev: i32 = 0;
            let _ = cu::cuCtxGetDevice(&mut dev as *mut _);
            dev as u32
        };
        if cur_dev != self.device_id {
            return Err(CudaMabError::DeviceMismatch {
                buf: self.device_id,
                current: cur_dev,
            });
        }

        if len == 0 || rows == 0 {
            return Err(CudaMabError::InvalidInput(
                "len and rows must be positive".into(),
            ));
        }
        if len > i32::MAX as usize || rows > i32::MAX as usize || first_valid > i32::MAX as usize {
            return Err(CudaMabError::InvalidInput(
                "arguments exceed kernel limits".into(),
            ));
        }

        let mut func: Function = self
            .module
            .get_function("mab_batch_from_prefix_sma_f32")
            .map_err(|_e| CudaMabError::MissingKernelSymbol {
                name: "mab_batch_from_prefix_sma_f32",
            })?;

        let block_x: u32 = 128;
        let grid_x = ((rows as u32) + block_x - 1) / block_x;
        let grid = GridSize::xyz(grid_x.max(1), 1, 1);
        let block = BlockSize::xyz(block_x, 1, 1);

        unsafe {
            let mut pcs_p = d_pref_close_sum.as_device_ptr().as_raw();
            let mut pcn_p = d_pref_close_nan.as_device_ptr().as_raw();
            let mut fast_p = d_fast_periods.as_device_ptr().as_raw();
            let mut slow_p = d_slow_periods.as_device_ptr().as_raw();
            let mut up_p = d_devups.as_device_ptr().as_raw();
            let mut dn_p = d_devdns.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut fv_i = first_valid as i32;
            let mut rows_i = rows as i32;
            let mut out_u = d_upper.as_device_ptr().as_raw();
            let mut out_m = d_middle.as_device_ptr().as_raw();
            let mut out_l = d_lower.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut pcs_p as *mut _ as *mut c_void,
                &mut pcn_p as *mut _ as *mut c_void,
                &mut fast_p as *mut _ as *mut c_void,
                &mut slow_p as *mut _ as *mut c_void,
                &mut up_p as *mut _ as *mut c_void,
                &mut dn_p as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut fv_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut out_u as *mut _ as *mut c_void,
                &mut out_m as *mut _ as *mut c_void,
                &mut out_l as *mut _ as *mut c_void,
            ];
            self.stream.launch(&mut func, grid, block, 0, args)?;
        }

        Ok(())
    }

    pub fn mab_batch_dev(
        &self,
        prices_f32: &[f32],
        sweep: &MabBatchRange,
    ) -> Result<(DeviceArrayF32Triplet, Vec<MabParams>), CudaMabError> {
        if prices_f32.is_empty() {
            return Err(CudaMabError::InvalidInput("empty input".into()));
        }
        let len = prices_f32.len();
        let first_valid = prices_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaMabError::InvalidInput("all values are NaN".into()))?;

        let combos = crate::indicators::mab::expand_grid(sweep)
            .map_err(|e| CudaMabError::InvalidInput(e.to_string()))?;
        if combos.is_empty() {
            return Err(CudaMabError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let rows = combos.len();
        let elem_count = rows
            .checked_mul(len)
            .ok_or_else(|| CudaMabError::InvalidInput("rows*len overflow".into()))?;
        let out_bytes = elem_count
            .checked_mul(3)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaMabError::InvalidInput("output byte size overflow".into()))?;
        let in_bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaMabError::InvalidInput("input byte size overflow".into()))?;
        let required = out_bytes
            .checked_add(in_bytes)
            .ok_or_else(|| CudaMabError::InvalidInput("total byte size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            if let Ok((free, _total)) = mem_get_info() {
                return Err(CudaMabError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaMabError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        let devups: Vec<f32> = combos.iter().map(|p| p.devup.unwrap() as f32).collect();
        let devdns: Vec<f32> = combos.iter().map(|p| p.devdn.unwrap() as f32).collect();

        let all_sma = combos.iter().all(|p| {
            p.fast_ma_type
                .as_deref()
                .unwrap_or("sma")
                .eq_ignore_ascii_case("sma")
                && p.slow_ma_type
                    .as_deref()
                    .unwrap_or("sma")
                    .eq_ignore_ascii_case("sma")
        });

        let p0 = &combos[0];
        let all_same_ma = combos.iter().all(|p| {
            p.fast_period == p0.fast_period
                && p.slow_period == p0.slow_period
                && p.fast_ma_type == p0.fast_ma_type
                && p.slow_ma_type == p0.slow_ma_type
        });

        let elems = elem_count;
        let mut d_upper: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }?;
        let mut d_middle: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }?;
        let mut d_lower: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }?;

        if all_sma && !all_same_ma {
            let (pcs, pcn) = Self::build_prefixes_single(prices_f32);
            let d_pcs = DeviceBuffer::from_slice(&pcs)?;
            let d_pcn = DeviceBuffer::from_slice(&pcn)?;

            let fast_periods: Vec<i32> = combos
                .iter()
                .map(|p| p.fast_period.unwrap_or(0) as i32)
                .collect();
            let slow_periods: Vec<i32> = combos
                .iter()
                .map(|p| p.slow_period.unwrap_or(0) as i32)
                .collect();
            let d_fast_periods = DeviceBuffer::from_slice(&fast_periods)?;
            let d_slow_periods = DeviceBuffer::from_slice(&slow_periods)?;
            let d_devups = DeviceBuffer::from_slice(&devups)?;
            let d_devdns = DeviceBuffer::from_slice(&devdns)?;

            self.mab_batch_device_sma(
                &d_pcs,
                &d_pcn,
                &d_fast_periods,
                &d_slow_periods,
                &d_devups,
                &d_devdns,
                len,
                first_valid,
                rows,
                &mut d_upper,
                &mut d_middle,
                &mut d_lower,
            )?;
            self.stream.synchronize()?;

            let trip = DeviceArrayF32Triplet {
                upper: DeviceArrayF32 {
                    buf: d_upper,
                    rows,
                    cols: len,
                },
                middle: DeviceArrayF32 {
                    buf: d_middle,
                    rows,
                    cols: len,
                },
                lower: DeviceArrayF32 {
                    buf: d_lower,
                    rows,
                    cols: len,
                },
            };
            return Ok((trip, combos));
        }

        if all_same_ma {
            let fast_ma_host = Self::compute_ma_host(
                p0.fast_ma_type.as_deref().unwrap_or("sma"),
                prices_f32,
                p0.fast_period.unwrap(),
            )?;
            let slow_ma_host = Self::compute_ma_host(
                p0.slow_ma_type.as_deref().unwrap_or("sma"),
                prices_f32,
                p0.slow_period.unwrap(),
            )?;
            let d_fast = DeviceBuffer::from_slice(&fast_ma_host)?;
            let d_slow = DeviceBuffer::from_slice(&slow_ma_host)?;

            let mut d_dev: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }?;

            let mut f_dev: Function =
                self.module
                    .get_function("mab_dev_from_ma_f32")
                    .map_err(|_e| CudaMabError::MissingKernelSymbol {
                        name: "mab_dev_from_ma_f32",
                    })?;

            unsafe {
                let mut fast_ptr = d_fast.as_device_ptr().as_raw();
                let mut slow_ptr = d_slow.as_device_ptr().as_raw();
                let mut fp_i = p0.fast_period.unwrap() as i32;
                let mut fv_i = first_valid as i32;
                let mut len_i = len as i32;
                let mut dev_ptr = d_dev.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut fast_ptr as *mut _ as *mut c_void,
                    &mut slow_ptr as *mut _ as *mut c_void,
                    &mut fp_i as *mut _ as *mut c_void,
                    &mut fv_i as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut dev_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(
                    &mut f_dev,
                    GridSize::xyz(1, 1, 1),
                    BlockSize::xyz(1, 1, 1),
                    0,
                    args,
                )?;
            }

            let h_ups = LockedBuffer::from_slice(&devups)?;
            let h_dns = LockedBuffer::from_slice(&devdns)?;
            let mut d_ups =
                unsafe { DeviceBuffer::<f32>::uninitialized_async(rows, &self.stream) }?;
            let mut d_dns =
                unsafe { DeviceBuffer::<f32>::uninitialized_async(rows, &self.stream) }?;
            unsafe {
                d_ups.async_copy_from(&h_ups, &self.stream)?;
                d_dns.async_copy_from(&h_dns, &self.stream)?;
            }

            let mut f_apply: Function = self
                .module
                .get_function("mab_apply_dev_shared_ma_batch_f32")
                .map_err(|_e| CudaMabError::MissingKernelSymbol {
                    name: "mab_apply_dev_shared_ma_batch_f32",
                })?;

            let block_x: u32 = 256;
            let grid_x = ((len as u32) + block_x - 1) / block_x;
            let grid = GridSize::xyz(grid_x.max(1), rows as u32, 1);
            let block = BlockSize::xyz(block_x, 1, 1);

            unsafe {
                let mut fast_ptr = d_fast.as_device_ptr().as_raw();
                let mut slow_ptr = d_slow.as_device_ptr().as_raw();
                let mut dev_ptr = d_dev.as_device_ptr().as_raw();
                let mut fp_i = p0.fast_period.unwrap() as i32;
                let mut sp_i = p0.slow_period.unwrap() as i32;
                let mut fv_i = first_valid as i32;
                let mut len_i = len as i32;
                let mut ups_ptr = d_ups.as_device_ptr().as_raw();
                let mut dns_ptr = d_dns.as_device_ptr().as_raw();
                let mut rows_i = rows as i32;
                let mut up_ptr = d_upper.as_device_ptr().as_raw();
                let mut mid_ptr = d_middle.as_device_ptr().as_raw();
                let mut lo_ptr = d_lower.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut fast_ptr as *mut _ as *mut c_void,
                    &mut slow_ptr as *mut _ as *mut c_void,
                    &mut dev_ptr as *mut _ as *mut c_void,
                    &mut fp_i as *mut _ as *mut c_void,
                    &mut sp_i as *mut _ as *mut c_void,
                    &mut fv_i as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut ups_ptr as *mut _ as *mut c_void,
                    &mut dns_ptr as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut up_ptr as *mut _ as *mut c_void,
                    &mut mid_ptr as *mut _ as *mut c_void,
                    &mut lo_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&mut f_apply, grid, block, 0, args)?;
            }
        } else {
            let mut f_row: Function = self
                .module
                .get_function("mab_single_row_from_ma_f32")
                .map_err(|_e| CudaMabError::MissingKernelSymbol {
                    name: "mab_single_row_from_ma_f32",
                })?;
            for (row, p) in combos.iter().enumerate() {
                let fast_ma_host = Self::compute_ma_host(
                    p.fast_ma_type.as_deref().unwrap_or("sma"),
                    prices_f32,
                    p.fast_period.unwrap(),
                )?;
                let slow_ma_host = Self::compute_ma_host(
                    p.slow_ma_type.as_deref().unwrap_or("sma"),
                    prices_f32,
                    p.slow_period.unwrap(),
                )?;
                let d_fast = DeviceBuffer::from_slice(&fast_ma_host)?;
                let d_slow = DeviceBuffer::from_slice(&slow_ma_host)?;

                let row_off = row * len;
                let mut up_row =
                    unsafe { d_upper.as_device_ptr().offset(row_off as isize).as_raw() };
                let mut mid_row =
                    unsafe { d_middle.as_device_ptr().offset(row_off as isize).as_raw() };
                let mut lo_row =
                    unsafe { d_lower.as_device_ptr().offset(row_off as isize).as_raw() };

                unsafe {
                    let mut fast_ptr = d_fast.as_device_ptr().as_raw();
                    let mut slow_ptr = d_slow.as_device_ptr().as_raw();
                    let mut fp_i = p.fast_period.unwrap() as i32;
                    let mut sp_i = p.slow_period.unwrap() as i32;
                    let mut fv_i = first_valid as i32;
                    let mut len_i = len as i32;
                    let mut upf = p.devup.unwrap() as f32;
                    let mut dnf = p.devdn.unwrap() as f32;
                    let args: &mut [*mut c_void] = &mut [
                        &mut fast_ptr as *mut _ as *mut c_void,
                        &mut slow_ptr as *mut _ as *mut c_void,
                        &mut fp_i as *mut _ as *mut c_void,
                        &mut sp_i as *mut _ as *mut c_void,
                        &mut fv_i as *mut _ as *mut c_void,
                        &mut len_i as *mut _ as *mut c_void,
                        &mut upf as *mut _ as *mut c_void,
                        &mut dnf as *mut _ as *mut c_void,
                        &mut up_row as *mut _ as *mut c_void,
                        &mut mid_row as *mut _ as *mut c_void,
                        &mut lo_row as *mut _ as *mut c_void,
                    ];
                    self.stream.launch(
                        &mut f_row,
                        GridSize::xyz(1, 1, 1),
                        BlockSize::xyz(1, 1, 1),
                        0,
                        args,
                    )?;
                }
            }
        }

        self.stream.synchronize()?;

        let trip = DeviceArrayF32Triplet {
            upper: DeviceArrayF32 {
                buf: d_upper,
                rows,
                cols: len,
            },
            middle: DeviceArrayF32 {
                buf: d_middle,
                rows,
                cols: len,
            },
            lower: DeviceArrayF32 {
                buf: d_lower,
                rows,
                cols: len,
            },
        };
        Ok((trip, combos))
    }

    pub fn mab_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &MabParams,
    ) -> Result<DeviceArrayF32Triplet, CudaMabError> {
        if cols == 0 || rows == 0 {
            return Err(CudaMabError::InvalidInput("invalid series dims".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaMabError::InvalidInput("cols*rows overflow".into()))?;
        if data_tm_f32.len() != expected {
            return Err(CudaMabError::InvalidInput(
                "time-major length mismatch".into(),
            ));
        }
        let fast = params.fast_period.unwrap_or(0);
        let slow = params.slow_period.unwrap_or(0);
        if fast == 0 || slow == 0 {
            return Err(CudaMabError::InvalidInput("periods must be >=1".into()));
        }

        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for r in 0..rows {
                if !data_tm_f32[r * cols + s].is_nan() {
                    fv = Some(r as i32);
                    break;
                }
            }
            let fv =
                fv.ok_or_else(|| CudaMabError::InvalidInput(format!("series {} all-NaN", s)))?;

            let need_total = (fast.max(slow) + fast - 1) as i32;
            if (rows as i32) - fv < need_total {
                return Err(CudaMabError::InvalidInput(format!(
                    "series {} insufficient valid tail for fast={}, slow={}",
                    s, fast, slow
                )));
            }
            first_valids[s] = fv;
        }

        let fast_type = params.fast_ma_type.as_deref().unwrap_or("sma");
        let slow_type = params.slow_ma_type.as_deref().unwrap_or("sma");

        let fast_tm_host =
            Self::compute_ma_host_time_major(fast_type, data_tm_f32, cols, rows, fast)?;
        let slow_tm_host =
            Self::compute_ma_host_time_major(slow_type, data_tm_f32, cols, rows, slow)?;
        let fast_dev = DeviceArrayF32 {
            buf: DeviceBuffer::from_slice(&fast_tm_host)?,
            rows,
            cols,
        };
        let slow_dev = DeviceArrayF32 {
            buf: DeviceBuffer::from_slice(&slow_tm_host)?,
            rows,
            cols,
        };

        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaMabError::InvalidInput("cols*rows overflow".into()))?;
        let mut d_upper: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }?;
        let mut d_middle: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }?;
        let mut d_lower: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }?;

        let mut func: Function = self
            .module
            .get_function("mab_many_series_one_param_time_major_f32")
            .map_err(|_e| CudaMabError::MissingKernelSymbol {
                name: "mab_many_series_one_param_time_major_f32",
            })?;

        let grid = GridSize::xyz(1, cols as u32, 1);
        let block = BlockSize::xyz(1, 1, 1);
        unsafe {
            let mut f_ptr = fast_dev.buf.as_device_ptr().as_raw();
            let mut s_ptr = slow_dev.buf.as_device_ptr().as_raw();
            let mut first_ptr = DeviceBuffer::from_slice(&first_valids)?
                .as_device_ptr()
                .as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut fp_i = fast as i32;
            let mut sp_i = slow as i32;
            let mut upf = params.devup.unwrap_or(1.0) as f32;
            let mut dnf = params.devdn.unwrap_or(1.0) as f32;
            let mut up_ptr = d_upper.as_device_ptr().as_raw();
            let mut mid_ptr = d_middle.as_device_ptr().as_raw();
            let mut lo_ptr = d_lower.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut f_ptr as *mut _ as *mut c_void,
                &mut s_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut fp_i as *mut _ as *mut c_void,
                &mut sp_i as *mut _ as *mut c_void,
                &mut upf as *mut _ as *mut c_void,
                &mut dnf as *mut _ as *mut c_void,
                &mut up_ptr as *mut _ as *mut c_void,
                &mut mid_ptr as *mut _ as *mut c_void,
                &mut lo_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&mut func, grid, block, 0, args)?;
        }

        self.stream.synchronize()?;
        Ok(DeviceArrayF32Triplet {
            upper: DeviceArrayF32 {
                buf: d_upper,
                rows,
                cols,
            },
            middle: DeviceArrayF32 {
                buf: d_middle,
                rows,
                cols,
            },
            lower: DeviceArrayF32 {
                buf: d_lower,
                rows,
                cols,
            },
        })
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "mab",
                "batch_dev",
                "mab_cuda_batch_dev",
                "60k_x_49combos",
                prep_mab_batch_box,
            )
            .with_inner_iters(4),
            CudaBenchScenario::new(
                "mab",
                "many_series_one_param",
                "mab_cuda_many_series_one_param",
                "128x1m",
                prep_mab_many_series_box,
            )
            .with_inner_iters(2),
        ]
    }

    struct MabBatchState {
        cuda: CudaMab,
        d_pcs: DeviceBuffer<f64>,
        d_pcn: DeviceBuffer<i32>,
        d_fast: DeviceBuffer<i32>,
        d_slow: DeviceBuffer<i32>,
        d_devups: DeviceBuffer<f32>,
        d_devdns: DeviceBuffer<f32>,
        d_up: DeviceBuffer<f32>,
        d_mid: DeviceBuffer<f32>,
        d_lo: DeviceBuffer<f32>,
        rows: usize,
        len: usize,
        first_valid: usize,
    }

    impl CudaBenchState for MabBatchState {
        fn launch(&mut self) {
            self.cuda
                .mab_batch_device_sma(
                    &self.d_pcs,
                    &self.d_pcn,
                    &self.d_fast,
                    &self.d_slow,
                    &self.d_devups,
                    &self.d_devdns,
                    self.len,
                    self.first_valid,
                    self.rows,
                    &mut self.d_up,
                    &mut self.d_mid,
                    &mut self.d_lo,
                )
                .expect("mab_batch_device_sma");
            self.cuda.stream.synchronize().unwrap();
        }
    }

    fn prep_mab_batch() -> MabBatchState {
        let cuda = CudaMab::new(0).expect("cuda mab");
        let len = 60_000usize;
        let mut price = vec![f32::NAN; len];
        for i in 10..len {
            let x = i as f32;
            price[i] = (x * 0.001).sin() + 0.001 * x;
        }
        let first_valid = price.iter().position(|v| !v.is_nan()).unwrap_or(0);
        let (pcs, pcn) = CudaMab::build_prefixes_single(&price);

        let sweep = MabBatchRange {
            fast_period: (10, 22, 2),
            slow_period: (50, 74, 4),
            devup: (1.0, 1.0, 0.0),
            devdn: (1.0, 1.0, 0.0),
            fast_ma_type: ("sma".into(), "sma".into(), "".into()),
            slow_ma_type: ("sma".into(), "sma".into(), "".into()),
        };
        let combos = crate::indicators::mab::expand_grid(&sweep).expect("expand mab grid");
        let rows = combos.len();
        assert_eq!(rows, 49, "unexpected MAB combo count");

        let fast_periods: Vec<i32> = combos
            .iter()
            .map(|p| p.fast_period.unwrap_or(0) as i32)
            .collect();
        let slow_periods: Vec<i32> = combos
            .iter()
            .map(|p| p.slow_period.unwrap_or(0) as i32)
            .collect();
        let devups: Vec<f32> = combos.iter().map(|p| p.devup.unwrap() as f32).collect();
        let devdns: Vec<f32> = combos.iter().map(|p| p.devdn.unwrap() as f32).collect();

        let d_pcs = DeviceBuffer::from_slice(&pcs).expect("upload pcs");
        let d_pcn = DeviceBuffer::from_slice(&pcn).expect("upload pcn");
        let d_fast = DeviceBuffer::from_slice(&fast_periods).expect("upload fast");
        let d_slow = DeviceBuffer::from_slice(&slow_periods).expect("upload slow");
        let d_devups = DeviceBuffer::from_slice(&devups).expect("upload devup");
        let d_devdns = DeviceBuffer::from_slice(&devdns).expect("upload devdn");

        let elems = rows * len;
        let d_up: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.unwrap();
        let d_mid: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.unwrap();
        let d_lo: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.unwrap();
        MabBatchState {
            cuda,
            d_pcs,
            d_pcn,
            d_fast,
            d_slow,
            d_devups,
            d_devdns,
            d_up,
            d_mid,
            d_lo,
            rows,
            len,
            first_valid,
        }
    }
    fn prep_mab_batch_box() -> Box<dyn CudaBenchState> {
        Box::new(prep_mab_batch())
    }

    struct MabManySeriesState {
        cuda: CudaMab,
        d_fast_tm: DeviceBuffer<f32>,
        d_slow_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        fast: usize,
        slow: usize,
        devup: f32,
        devdn: f32,
        d_up: DeviceBuffer<f32>,
        d_mid: DeviceBuffer<f32>,
        d_lo: DeviceBuffer<f32>,
    }
    impl CudaBenchState for MabManySeriesState {
        fn launch(&mut self) {
            let mut func: Function = self
                .cuda
                .module
                .get_function("mab_many_series_one_param_time_major_f32")
                .expect("mab_many_series_one_param_time_major_f32");

            let grid = GridSize::xyz(1, self.cols as u32, 1);
            let block = BlockSize::xyz(1, 1, 1);
            unsafe {
                let mut f_ptr = self.d_fast_tm.as_device_ptr().as_raw();
                let mut s_ptr = self.d_slow_tm.as_device_ptr().as_raw();
                let mut first_ptr = self.d_first_valids.as_device_ptr().as_raw();
                let mut cols_i = self.cols as i32;
                let mut rows_i = self.rows as i32;
                let mut fp_i = self.fast as i32;
                let mut sp_i = self.slow as i32;
                let mut upf = self.devup;
                let mut dnf = self.devdn;
                let mut up_ptr = self.d_up.as_device_ptr().as_raw();
                let mut mid_ptr = self.d_mid.as_device_ptr().as_raw();
                let mut lo_ptr = self.d_lo.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut f_ptr as *mut _ as *mut c_void,
                    &mut s_ptr as *mut _ as *mut c_void,
                    &mut first_ptr as *mut _ as *mut c_void,
                    &mut cols_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut fp_i as *mut _ as *mut c_void,
                    &mut sp_i as *mut _ as *mut c_void,
                    &mut upf as *mut _ as *mut c_void,
                    &mut dnf as *mut _ as *mut c_void,
                    &mut up_ptr as *mut _ as *mut c_void,
                    &mut mid_ptr as *mut _ as *mut c_void,
                    &mut lo_ptr as *mut _ as *mut c_void,
                ];
                self.cuda
                    .stream
                    .launch(&mut func, grid, block, 0, args)
                    .unwrap();
            }

            self.cuda.stream.synchronize().unwrap();
        }
    }
    fn prep_mab_many_series() -> MabManySeriesState {
        let cuda = CudaMab::new(0).expect("cuda mab");
        let cols = 128usize;
        let rows = 1_000_000usize;
        let mut tm = vec![f32::NAN; cols * rows];
        for s in 0..cols {
            for r in s..rows {
                let x = (r as f32) + 0.1 * (s as f32);
                tm[r * cols + s] = (x * 0.002).sin() + 0.0005 * x;
            }
        }
        let p = MabParams {
            fast_period: Some(10),
            slow_period: Some(50),
            devup: Some(1.0),
            devdn: Some(1.0),
            fast_ma_type: Some("sma".into()),
            slow_ma_type: Some("sma".into()),
        };

        let fast = p.fast_period.unwrap_or(0);
        let slow = p.slow_period.unwrap_or(0);
        let devup = p.devup.unwrap_or(1.0) as f32;
        let devdn = p.devdn.unwrap_or(1.0) as f32;

        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for r in 0..rows {
                if !tm[r * cols + s].is_nan() {
                    fv = Some(r as i32);
                    break;
                }
            }
            first_valids[s] = fv.unwrap_or(0);
        }

        let fast_type = p.fast_ma_type.as_deref().unwrap_or("sma");
        let slow_type = p.slow_ma_type.as_deref().unwrap_or("sma");
        let fast_tm_host =
            CudaMab::compute_ma_host_time_major(fast_type, &tm, cols, rows, fast).unwrap();
        let slow_tm_host =
            CudaMab::compute_ma_host_time_major(slow_type, &tm, cols, rows, slow).unwrap();

        let d_fast_tm = DeviceBuffer::from_slice(&fast_tm_host).expect("d_fast_tm");
        let d_slow_tm = DeviceBuffer::from_slice(&slow_tm_host).expect("d_slow_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");

        let elems = cols * rows;
        let d_up: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.unwrap();
        let d_mid: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.unwrap();
        let d_lo: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.unwrap();
        cuda.stream.synchronize().unwrap();

        MabManySeriesState {
            cuda,
            d_fast_tm,
            d_slow_tm,
            d_first_valids,
            cols,
            rows,
            fast,
            slow,
            devup,
            devdn,
            d_up,
            d_mid,
            d_lo,
        }
    }
    fn prep_mab_many_series_box() -> Box<dyn CudaBenchState> {
        Box::new(prep_mab_many_series())
    }
}
