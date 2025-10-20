#![cfg(feature = "cuda")]

//! CUDA wrapper for VPCI (Volume Price Confirmation Index).
//!
//! Math pattern: prefix-sum/rational, like VI/VWMA. The batch path (one series × many params)
//! and the many-series path (time-major, one param) both precompute prefix sums on the host
//! for close, volume, and close*volume, and upload them once. Warmup/NaN semantics match the
//! scalar Rust implementation exactly: warm = first_valid + long - 1; indices < warm are NaN.

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::vpci::{VpciBatchRange, VpciParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaVpciError {
    Cuda(String),
    InvalidInput(String),
}
impl fmt::Display for CudaVpciError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaVpciError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaVpciError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl std::error::Error for CudaVpciError {}

// Pair of VRAM-resident arrays (VPCI and VPCIS)
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
pub struct CudaVpciPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

pub struct CudaVpci {
    module: Module,
    stream: Stream,
    _context: Context,
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
        cust::init(CudaFlags::empty()).map_err(|e| CudaVpciError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaVpciError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaVpciError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/vpci_kernel.ptx"));
        let jit = [
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, &jit)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaVpciError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaVpciError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
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
        self.stream
            .synchronize()
            .map_err(|e| CudaVpciError::Cuda(e.to_string()))
    }

    #[inline]
    fn mem_ok(bytes: usize, headroom: usize) -> bool {
        if std::env::var("CUDA_MEM_CHECK")
            .ok()
            .filter(|v| v == "0" || v.eq_ignore_ascii_case("false"))
            .is_some()
        {
            return true;
        }
        mem_get_info()
            .map(|(free, _)| bytes.saturating_add(headroom) <= free)
            .unwrap_or(true)
    }

    // --------------- Prefix sums (single series) ---------------
    fn build_prefix_single(
        &self,
        close: &[f32],
        volume: &[f32],
    ) -> Result<(usize, Vec<f64>, Vec<f64>, Vec<f64>), CudaVpciError> {
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
        let mut pfx_c = vec![0.0f64; n];
        let mut pfx_v = vec![0.0f64; n];
        let mut pfx_cv = vec![0.0f64; n];
        pfx_c[first] = close[first] as f64;
        pfx_v[first] = volume[first] as f64;
        pfx_cv[first] = (close[first] as f64) * (volume[first] as f64);
        for i in (first + 1)..n {
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
            pfx_c[i] = pfx_c[i - 1] + c;
            pfx_v[i] = pfx_v[i - 1] + v;
            pfx_cv[i] = pfx_cv[i - 1] + c * v;
        }
        Ok((first, pfx_c, pfx_v, pfx_cv))
    }

    // --------------- Prefix sums (many series, time-major) ---------------
    fn build_prefix_tm(
        &self,
        close_tm: &[f32],
        volume_tm: &[f32],
        cols: usize,
        rows: usize,
    ) -> Result<(Vec<i32>, Vec<f64>, Vec<f64>, Vec<f64>), CudaVpciError> {
        if close_tm.len() != volume_tm.len() {
            return Err(CudaVpciError::InvalidInput("length mismatch".into()));
        }
        if cols == 0 || rows == 0 {
            return Err(CudaVpciError::InvalidInput("invalid dims".into()));
        }
        if close_tm.len() != cols * rows {
            return Err(CudaVpciError::InvalidInput(
                "dims do not match data length".into(),
            ));
        }

        let mut first_valids = vec![-1i32; cols];
        let mut pfx_c = vec![0.0f64; rows * cols];
        let mut pfx_v = vec![0.0f64; rows * cols];
        let mut pfx_cv = vec![0.0f64; rows * cols];

        for s in 0..cols {
            // find column first valid where both close and volume are finite
            let mut first = None;
            for r in 0..rows {
                let idx = r * cols + s;
                if close_tm[idx].is_finite() && volume_tm[idx].is_finite() {
                    first = Some(r);
                    break;
                }
            }
            if let Some(fv) = first {
                first_valids[s] = fv as i32;
                let base = fv * cols + s;
                pfx_c[base] = close_tm[base] as f64;
                pfx_v[base] = volume_tm[base] as f64;
                pfx_cv[base] = (close_tm[base] as f64) * (volume_tm[base] as f64);
                for r in (fv + 1)..rows {
                    let idx = r * cols + s;
                    let prev = (r - 1) * cols + s;
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
                    pfx_c[idx] = pfx_c[prev] + c;
                    pfx_v[idx] = pfx_v[prev] + v;
                    pfx_cv[idx] = pfx_cv[prev] + c * v;
                }
            } else {
                // all NaN: leave prefix zeros, keep first_valid = -1
            }
        }

        Ok((first_valids, pfx_c, pfx_v, pfx_cv))
    }

    // --------------- Batch entry ---------------
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

        // Expand grid locally
        fn axis_usize((s, e, st): (usize, usize, usize)) -> Vec<usize> {
            if st == 0 || s == e {
                vec![s]
            } else {
                (s..=e).step_by(st).collect()
            }
        }
        let shorts = axis_usize(sweep.short_range);
        let longs = axis_usize(sweep.long_range);
        let mut combos = Vec::with_capacity(shorts.len() * longs.len());
        for &s in &shorts {
            for &l in &longs {
                combos.push(VpciParams {
                    short_range: Some(s),
                    long_range: Some(l),
                });
            }
        }
        if combos.is_empty() {
            return Err(CudaVpciError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let max_long = combos.iter().map(|p| p.long_range.unwrap()).max().unwrap();

        let (first_valid, pfx_c, pfx_v, pfx_cv) =
            self.build_prefix_single(close_f32, volume_f32)?;
        if len - first_valid < max_long {
            return Err(CudaVpciError::InvalidInput(
                "insufficient valid data after first_valid".into(),
            ));
        }

        // VRAM estimate: 3 * prefix (f64) + volume (f32) + params (i32 * 2) + 2 * outputs (f32)
        let rows = combos.len();
        let bytes = 3 * len * std::mem::size_of::<f64>()
            + len * std::mem::size_of::<f32>()
            + rows * 2 * std::mem::size_of::<i32>()
            + 2 * rows * len * std::mem::size_of::<f32>();
        let headroom = std::env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        if !Self::mem_ok(bytes, headroom) {
            return Err(CudaVpciError::InvalidInput(
                "insufficient VRAM for VPCI batch".into(),
            ));
        }

        // Upload inputs
        let h_pfx_c =
            LockedBuffer::from_slice(&pfx_c).map_err(|e| CudaVpciError::Cuda(e.to_string()))?;
        let h_pfx_v =
            LockedBuffer::from_slice(&pfx_v).map_err(|e| CudaVpciError::Cuda(e.to_string()))?;
        let h_pfx_cv =
            LockedBuffer::from_slice(&pfx_cv).map_err(|e| CudaVpciError::Cuda(e.to_string()))?;
        let d_pfx_c = unsafe { DeviceBuffer::from_slice_async(h_pfx_c.as_slice(), &self.stream) }
            .map_err(|e| CudaVpciError::Cuda(e.to_string()))?;
        let d_pfx_v = unsafe { DeviceBuffer::from_slice_async(h_pfx_v.as_slice(), &self.stream) }
            .map_err(|e| CudaVpciError::Cuda(e.to_string()))?;
        let d_pfx_cv = unsafe { DeviceBuffer::from_slice_async(h_pfx_cv.as_slice(), &self.stream) }
            .map_err(|e| CudaVpciError::Cuda(e.to_string()))?;
        let d_vol = unsafe { DeviceBuffer::from_slice_async(volume_f32, &self.stream) }
            .map_err(|e| CudaVpciError::Cuda(e.to_string()))?;

        let shorts_i32: Vec<i32> = combos
            .iter()
            .map(|p| p.short_range.unwrap() as i32)
            .collect();
        let longs_i32: Vec<i32> = combos
            .iter()
            .map(|p| p.long_range.unwrap() as i32)
            .collect();
        let d_shorts = unsafe { DeviceBuffer::from_slice_async(&shorts_i32, &self.stream) }
            .map_err(|e| CudaVpciError::Cuda(e.to_string()))?;
        let d_longs = unsafe { DeviceBuffer::from_slice_async(&longs_i32, &self.stream) }
            .map_err(|e| CudaVpciError::Cuda(e.to_string()))?;

        // Outputs
        let mut d_vpci: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(rows * len) }
            .map_err(|e| CudaVpciError::Cuda(e.to_string()))?;
        let mut d_vpcis: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(rows * len) }
            .map_err(|e| CudaVpciError::Cuda(e.to_string()))?;

        // Launch
        let func = self
            .module
            .get_function("vpci_batch_f32")
            .map_err(|e| CudaVpciError::Cuda(e.to_string()))?;
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            _ => 256,
        };
        let grid_x = ((rows as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
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

            let mut args: [*mut c_void; 12] = [
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
                std::ptr::null_mut(), // padding (unused)
            ];

            self.stream
                .launch(&func, grid, block, 0, &mut args[..])
                .map_err(|e| CudaVpciError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaVpciError::Cuda(e.to_string()))?;

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

    // --------------- Many-series × one-param (time-major) ---------------
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
        if close_tm_f32.len() != rows * cols || volume_tm_f32.len() != rows * cols {
            return Err(CudaVpciError::InvalidInput(
                "dims do not match data length".into(),
            ));
        }

        let (first_valids, pfx_c, pfx_v, pfx_cv) =
            self.build_prefix_tm(close_tm_f32, volume_tm_f32, cols, rows)?;

        // Upload
        let h_pfx_c =
            LockedBuffer::from_slice(&pfx_c).map_err(|e| CudaVpciError::Cuda(e.to_string()))?;
        let h_pfx_v =
            LockedBuffer::from_slice(&pfx_v).map_err(|e| CudaVpciError::Cuda(e.to_string()))?;
        let h_pfx_cv =
            LockedBuffer::from_slice(&pfx_cv).map_err(|e| CudaVpciError::Cuda(e.to_string()))?;
        let h_firsts = LockedBuffer::from_slice(&first_valids)
            .map_err(|e| CudaVpciError::Cuda(e.to_string()))?;
        let d_pfx_c = unsafe { DeviceBuffer::from_slice_async(h_pfx_c.as_slice(), &self.stream) }
            .map_err(|e| CudaVpciError::Cuda(e.to_string()))?;
        let d_pfx_v = unsafe { DeviceBuffer::from_slice_async(h_pfx_v.as_slice(), &self.stream) }
            .map_err(|e| CudaVpciError::Cuda(e.to_string()))?;
        let d_pfx_cv = unsafe { DeviceBuffer::from_slice_async(h_pfx_cv.as_slice(), &self.stream) }
            .map_err(|e| CudaVpciError::Cuda(e.to_string()))?;
        let d_firsts = unsafe { DeviceBuffer::from_slice_async(h_firsts.as_slice(), &self.stream) }
            .map_err(|e| CudaVpciError::Cuda(e.to_string()))?;
        let d_vol = unsafe { DeviceBuffer::from_slice_async(volume_tm_f32, &self.stream) }
            .map_err(|e| CudaVpciError::Cuda(e.to_string()))?;

        // Outputs
        let total = rows * cols;
        let mut d_vpci: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }
            .map_err(|e| CudaVpciError::Cuda(e.to_string()))?;
        let mut d_vpcis: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }
            .map_err(|e| CudaVpciError::Cuda(e.to_string()))?;

        // Launch (X across series)
        let func = self
            .module
            .get_function("vpci_many_series_one_param_f32")
            .map_err(|e| CudaVpciError::Cuda(e.to_string()))?;
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            _ => 256,
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
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
            let mut args: [*mut c_void; 12] = [
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
                std::ptr::null_mut(),
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args[..])
                .map_err(|e| CudaVpciError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaVpciError::Cuda(e.to_string()))?;

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

// ---------- Bench profiles ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series(rows: usize) -> usize {
        // 2 inputs (C,V) + 2 outputs (VPCI,VPCIS) + 3 prefixes (f64) + params
        2 * ONE_SERIES_LEN * std::mem::size_of::<f32>()
            + 2 * rows * ONE_SERIES_LEN * std::mem::size_of::<f32>()
            + 3 * ONE_SERIES_LEN * std::mem::size_of::<f64>()
            + rows * 2 * std::mem::size_of::<i32>()
            + 64 * 1024 * 1024
    }

    struct BatchState {
        cuda: CudaVpci,
        c: Vec<f32>,
        v: Vec<f32>,
        sweep: VpciBatchRange,
    }
    impl CudaBenchState for BatchState {
        fn launch(&mut self) {
            let _ = self.cuda.vpci_batch_dev(&self.c, &self.v, &self.sweep);
        }
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let mut v = Vec::new();
        v.push(
            CudaBenchScenario::new(
                "vpci",
                "one_series_many_params",
                "vpci_batch",
                "vpci/batch/1e6",
                || {
                    let mut close = gen_series(ONE_SERIES_LEN);
                    let mut vol = gen_series(ONE_SERIES_LEN);
                    // Ensure both finite from some point
                    for i in 0..1024 {
                        close[i] = f32::NAN;
                        vol[i] = f32::NAN;
                    }
                    Box::new(BatchState {
                        cuda: CudaVpci::new(0).unwrap(),
                        c: close,
                        v: vol,
                        sweep: VpciBatchRange {
                            short_range: (5, 20, 1),
                            long_range: (25, 60, 5),
                        },
                    })
                },
            )
            .with_mem_required(bytes_one_series(((60 - 25) / 5 + 1) * ((20 - 5) / 1 + 1))),
        );
        v
    }
}
