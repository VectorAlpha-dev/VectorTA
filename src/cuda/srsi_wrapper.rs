//! CUDA wrapper for SRSI (Stochastic RSI)
//!
//! Category: Hybrid (recurrence + window extrema + SMAs)
//! - Batch (one series × many params): host-precompute RSI once per distinct
//!   `rsi_period` and launch per-group kernels over shared RSI (parity with
//!   scalar batch which reuses RSI cache). FP32 on device.
//! - Many-series × one-param (time-major): per-series Wilder RSI + window
//!   deques for extrema with SMA smoothing. FP32.
//!
//! Parity items mirrored from ALMA/CWMA:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/srsi_kernel.ptx")) with
//!   JIT opts DetermineTargetFromContext + OptLevel O2 (graceful fallbacks).
//! - Stream NON_BLOCKING; VRAM check with ~64MB headroom.
//! - Grid.x chunking to <= 65_535.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::rsi::{rsi, RsiInput, RsiParams};
use crate::indicators::srsi::{expand_grid_srsi, SrsiBatchRange, SrsiParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::error::CudaError;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::collections::BTreeMap;
use std::env;
use std::ffi::c_void;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaSrsiError {
    #[error(transparent)]
    Cuda(#[from] CudaError),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error(
        "launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})"
    )]
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

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaSrsiPolicy {
    pub batch_block_x: Option<u32>,
    pub many_block_x: Option<u32>,
}

pub struct DeviceSrsiPair {
    pub k: DeviceArrayF32,
    pub d: DeviceArrayF32,
}

pub struct CudaSrsi {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaSrsiPolicy,
    max_grid_x: u32,
}

impl CudaSrsi {
    pub fn new(device_id: usize) -> Result<Self, CudaSrsiError> {
        cust::init(CudaFlags::empty())?;
        let dev = Device::get_device(device_id as u32)?;
        let ctx = Arc::new(Context::new(dev)?);
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/srsi_kernel.ptx"));
        let jit = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        // Query true 1D grid X cap for chunking (often 2^31-1 on modern GPUs)
        let max_grid_x = dev
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(65_535) as u32;
        Ok(Self {
            module,
            stream,
            _context: ctx,
            device_id: device_id as u32,
            policy: CudaSrsiPolicy::default(),
            max_grid_x,
        })
    }

    #[inline]
    pub fn set_policy(&mut self, p: CudaSrsiPolicy) { self.policy = p; }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> {
        self._context.clone()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaSrsiError> {
        self.stream.synchronize().map_err(Into::into)
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaSrsiError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        match mem_get_info() {
            Ok((free, _)) => {
                if required_bytes.saturating_add(headroom_bytes) <= free {
                    Ok(())
                } else {
                    Err(CudaSrsiError::OutOfMemory {
                        required: required_bytes,
                        free,
                        headroom: headroom_bytes,
                    })
                }
            }
            Err(_) => Ok(()),
        }
    }

    #[inline]
    fn validate_launch_dims(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
    ) -> Result<(), CudaSrsiError> {
        let dev = Device::get_device(self.device_id)?;
        let max_gx = dev.get_attribute(DeviceAttribute::MaxGridDimX)? as u32;
        let max_gy = dev.get_attribute(DeviceAttribute::MaxGridDimY)? as u32;
        let max_gz = dev.get_attribute(DeviceAttribute::MaxGridDimZ)? as u32;
        let max_bx = dev.get_attribute(DeviceAttribute::MaxBlockDimX)? as u32;
        let max_by = dev.get_attribute(DeviceAttribute::MaxBlockDimY)? as u32;
        let max_bz = dev.get_attribute(DeviceAttribute::MaxBlockDimZ)? as u32;
        let (gx, gy, gz) = grid;
        let (bx, by, bz) = block;
        if gx == 0 || gy == 0 || gz == 0 || bx == 0 || by == 0 || bz == 0 {
            return Err(CudaSrsiError::InvalidInput(
                "zero-sized grid or block".into(),
            ));
        }
        if gx > max_gx || gy > max_gy || gz > max_gz || bx > max_bx || by > max_by || bz > max_bz
        {
            return Err(CudaSrsiError::LaunchConfigTooLarge {
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

    // --------- Batch (one series × many params) ---------
    pub fn srsi_batch_dev(
        &self,
        prices_f32: &[f32],
        sweep: &SrsiBatchRange,
    ) -> Result<(DeviceSrsiPair, Vec<SrsiParams>), CudaSrsiError> {
        let len = prices_f32.len();
        if len == 0 {
            return Err(CudaSrsiError::InvalidInput("empty series".into()));
        }
        let first_valid = (0..len)
            .find(|&i| !prices_f32[i].is_nan())
            .ok_or_else(|| CudaSrsiError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid_srsi(sweep)
            .map_err(|e| CudaSrsiError::InvalidInput(e.to_string()))?;
        if combos.is_empty() {
            return Err(CudaSrsiError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let max_need = combos
            .iter()
            .map(|c| {
                c.rsi_period
                    .unwrap()
                    .max(c.stoch_period.unwrap())
                    .max(c.k.unwrap())
                    .max(c.d.unwrap())
            })
            .max()
            .unwrap();
        if len - first_valid < max_need {
            return Err(CudaSrsiError::InvalidInput(
                "not enough valid data".into(),
            ));
        }

        // VRAM estimate (inputs + outputs + params + headroom)
        let elem_f32 = std::mem::size_of::<f32>();
        let elem_i32 = std::mem::size_of::<i32>();
        let in_bytes = len
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaSrsiError::InvalidInput("size overflow in input bytes".into()))?;
        let params_elems = combos
            .len()
            .checked_mul(3)
            .ok_or_else(|| CudaSrsiError::InvalidInput("size overflow in params elems".into()))?;
        let params_bytes = params_elems
            .checked_mul(elem_i32)
            .ok_or_else(|| CudaSrsiError::InvalidInput("size overflow in params bytes".into()))?;
        let out_elems = combos
            .len()
            .checked_mul(len)
            .ok_or_else(|| CudaSrsiError::InvalidInput("rows*cols overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(elem_f32 * 2)
            .ok_or_else(|| CudaSrsiError::InvalidInput("size overflow in output bytes".into()))?;
        let logical = in_bytes
            .checked_add(params_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaSrsiError::InvalidInput("total VRAM size overflow".into()))?;
        let headroom = 64usize * 1024 * 1024;
        let required = logical;
        Self::will_fit(required, headroom)?;
        // Group by rsi_period but preserve original row order
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, p) in combos.iter().enumerate() {
            groups.entry(p.rsi_period.unwrap()).or_default().push(i);
        }
        // Allocate outputs for all rows in original order
        let total = len
            .checked_mul(combos.len())
            .ok_or_else(|| CudaSrsiError::InvalidInput("rows*cols overflow".into()))?;
        let mut d_k: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total)? };
        let mut d_d: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total)? };

        // Build prices_f64 once (host RSI source)
        let prices_f64: Vec<f64> = prices_f32.iter().map(|&v| v as f64).collect();

        // Cache kernel handle & limits
        let func = self
            .module
            .get_function("srsi_batch_f32")
            .map_err(|_| CudaSrsiError::MissingKernelSymbol {
                name: "srsi_batch_f32",
            })?;
        let block_x = self.policy.batch_block_x.unwrap_or(128).min(1024);
        let grid_cap = self.max_grid_x.max(1);

        // Keep allocations alive until after sync
        let mut keep_alive: Vec<(DeviceBuffer<f32>, DeviceBuffer<i32>, DeviceBuffer<i32>, DeviceBuffer<i32>)> = Vec::new();

        for (rp, idxs) in groups {
            // Build RSI for this rp
            let rsi_out = rsi(&RsiInput::from_slice(&prices_f64, RsiParams { period: Some(rp) }))
                .map_err(|e| CudaSrsiError::InvalidInput(e.to_string()))?;
            let rsi_f32: Vec<f32> = rsi_out.values.into_iter().map(|v| v as f32).collect();
            let d_rsi = DeviceBuffer::from_slice(&rsi_f32)?;

            // Per-group parameter arrays following original index order
            let mut sp: Vec<i32> = Vec::with_capacity(idxs.len());
            let mut kp: Vec<i32> = Vec::with_capacity(idxs.len());
            let mut dp: Vec<i32> = Vec::with_capacity(idxs.len());
            let mut max_sp = 0usize;
            let mut max_k = 0usize;
            let mut max_d = 0usize;
            for &row in &idxs {
                let p = &combos[row];
                
                let s = p.stoch_period.unwrap();
                let k = p.k.unwrap();
                let d = p.d.unwrap();
                sp.push(s as i32);
                kp.push(k as i32);
                dp.push(d as i32);
                max_sp = max_sp.max(s);
                max_k = max_k.max(k);
                max_d = max_d.max(d);
            }
            let d_sp = DeviceBuffer::from_slice(&sp)?;
            let d_kp = DeviceBuffer::from_slice(&kp)?;
            let d_dp = DeviceBuffer::from_slice(&dp)?;

            // Dynamic shared memory per block
            let smem_bytes = (2 * max_sp * std::mem::size_of::<i32>()
                + (2 * max_sp + max_k + max_d) * std::mem::size_of::<f32>())
                as u32;

            // In current expand order, idxs are contiguous; use group_start and chunk by grid cap
            let group_start = *idxs.first().expect("group start");
            let mut base = 0usize;
            while base < idxs.len() {
                let chunk = (idxs.len() - base).min(grid_cap as usize);
                let gx = chunk as u32;
                self.validate_launch_dims((gx, 1, 1), (block_x, 1, 1))?;
                unsafe {
                    let mut rsi_ptr = d_rsi.as_device_ptr().as_raw() as u64;
                    let mut sp_ptr  = d_sp.as_device_ptr().as_raw().wrapping_add((base * std::mem::size_of::<i32>()) as u64);
                    let mut kp_ptr  = d_kp.as_device_ptr().as_raw().wrapping_add((base * std::mem::size_of::<i32>()) as u64);
                    let mut dp_ptr  = d_dp.as_device_ptr().as_raw().wrapping_add((base * std::mem::size_of::<i32>()) as u64);
                    let mut len_i = len as i32; let mut first_i = first_valid as i32; let mut rp_i = rp as i32; let mut n_i = chunk as i32;

                    let row_byte_off = ((group_start + base) * len * std::mem::size_of::<f32>()) as u64;
                    let mut out_k_ptr = d_k.as_device_ptr().as_raw().wrapping_add(row_byte_off);
                    let mut out_d_ptr = d_d.as_device_ptr().as_raw().wrapping_add(row_byte_off);

                    let mut args: [*mut c_void; 10] = [
                        &mut rsi_ptr as *mut _ as *mut c_void,
                        &mut sp_ptr  as *mut _ as *mut c_void,
                        &mut kp_ptr  as *mut _ as *mut c_void,
                        &mut dp_ptr  as *mut _ as *mut c_void,
                        &mut len_i   as *mut _ as *mut c_void,
                        &mut first_i as *mut _ as *mut c_void,
                        &mut rp_i    as *mut _ as *mut c_void,
                        &mut n_i     as *mut _ as *mut c_void,
                        &mut out_k_ptr as *mut _ as *mut c_void,
                        &mut out_d_ptr as *mut _ as *mut c_void,
                    ];
                    self.stream
                        .launch(
                            &func,
                            GridSize::x(chunk as u32),
                            BlockSize::x(block_x),
                            smem_bytes,
                            &mut args,
                        )?;
                }
                base += chunk;
            }

            keep_alive.push((d_rsi, d_sp, d_kp, d_dp));
        }

        // One sync for all groups
        self.stream.synchronize()?;

        Ok((
            DeviceSrsiPair {
                k: DeviceArrayF32 {
                    buf: d_k,
                    rows: combos.len(),
                    cols: len,
                },
                d: DeviceArrayF32 {
                    buf: d_d,
                    rows: combos.len(),
                    cols: len,
                },
            },
            combos,
        ))
    }

    // --------- Many-series, one param (time-major) ---------
    pub fn srsi_many_series_one_param_time_major_dev(
        &self,
        prices_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &SrsiParams,
    ) -> Result<DeviceSrsiPair, CudaSrsiError> {
        if cols == 0 || rows == 0 {
            return Err(CudaSrsiError::InvalidInput("empty matrix".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaSrsiError::InvalidInput("rows*cols overflow".into()))?;
        if prices_tm.len() != expected {
            return Err(CudaSrsiError::InvalidInput("size mismatch".into()));
        }
        let rp = params.rsi_period.unwrap_or(14) as i32;
        let sp = params.stoch_period.unwrap_or(14) as i32;
        let kp = params.k.unwrap_or(3) as i32;
        let dp = params.d.unwrap_or(3) as i32;
        if rp <= 0 || sp <= 0 || kp <= 0 || dp <= 0 {
            return Err(CudaSrsiError::InvalidInput(
                "non-positive periods".into(),
            ));
        }

        // Build first_valid per series on host
        let mut firsts = vec![rows as i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let v = prices_tm[t * cols + s];
                if v == v {
                    firsts[s] = t as i32;
                    break;
                }
            }
        }

        // VRAM estimate
        let elem_f32 = std::mem::size_of::<f32>();
        let elem_i32 = std::mem::size_of::<i32>();
        let n = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaSrsiError::InvalidInput("rows*cols overflow".into()))?;
        let in_bytes = n
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaSrsiError::InvalidInput("size overflow in input bytes".into()))?;
        let out_bytes = n
            .checked_mul(elem_f32 * 2)
            .ok_or_else(|| CudaSrsiError::InvalidInput("size overflow in output bytes".into()))?;
        let firsts_bytes = cols
            .checked_mul(elem_i32)
            .ok_or_else(|| CudaSrsiError::InvalidInput("size overflow in index bytes".into()))?;
        let logical = in_bytes
            .checked_add(out_bytes)
            .and_then(|x| x.checked_add(firsts_bytes))
            .ok_or_else(|| CudaSrsiError::InvalidInput("total VRAM size overflow".into()))?;
        let headroom = 64usize * 1024 * 1024;
        let required = logical;
        Self::will_fit(required, headroom)?;

        let d_prices = DeviceBuffer::from_slice(prices_tm)?;
        let d_firsts = DeviceBuffer::from_slice(&firsts)?;
        let total = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaSrsiError::InvalidInput("rows*cols overflow".into()))?;
        let mut d_k: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total)? };
        let mut d_d: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total)? };

        let func = self
            .module
            .get_function("srsi_many_series_one_param_f32")
            .map_err(|_| CudaSrsiError::MissingKernelSymbol {
                name: "srsi_many_series_one_param_f32",
            })?;
        let block_x = self.policy.many_block_x.unwrap_or(128).min(1024);
        let grid_cap = self.max_grid_x.max(1);
        let smem_bytes = (2 * (sp as usize) * std::mem::size_of::<i32>()
            + (2 * (sp as usize) + (kp as usize) + (dp as usize)) * std::mem::size_of::<f32>())
            as u32;

        let mut cols_done = 0usize;
        while cols_done < cols {
            let chunk_cols = (cols - cols_done).min(grid_cap as usize);
            let gx = chunk_cols as u32;
            self.validate_launch_dims((gx, 1, 1), (block_x, 1, 1))?;
            let mut prices_ptr = d_prices.as_device_ptr().as_raw().wrapping_add((cols_done * std::mem::size_of::<f32>()) as u64);
            let mut cols_i = cols as i32; let mut rows_i = rows as i32;
            let mut rp_i = rp; let mut sp_i = sp; let mut kp_i = kp; let mut dp_i = dp;
            let mut first_ptr = d_firsts.as_device_ptr().as_raw().wrapping_add((cols_done * std::mem::size_of::<i32>()) as u64);
            let mut k_ptr = d_k.as_device_ptr().as_raw().wrapping_add((cols_done * std::mem::size_of::<f32>()) as u64);
            let mut d_ptr = d_d.as_device_ptr().as_raw().wrapping_add((cols_done * std::mem::size_of::<f32>()) as u64);
            let mut args: [*mut c_void; 10] = [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut rp_i as *mut _ as *mut c_void,
                &mut sp_i as *mut _ as *mut c_void,
                &mut kp_i as *mut _ as *mut c_void,
                &mut dp_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut k_ptr as *mut _ as *mut c_void,
                &mut d_ptr as *mut _ as *mut c_void,
            ];
            unsafe {
                self.stream
                    .launch(
                        &func,
                        GridSize::x(chunk_cols as u32),
                        BlockSize::x(block_x),
                        smem_bytes,
                        &args,
                    )?;
            }
            cols_done += chunk_cols;
        }
        self.stream.synchronize()?;

        Ok(DeviceSrsiPair {
            k: DeviceArrayF32 { buf: d_k, rows, cols },
            d: DeviceArrayF32 { buf: d_d, rows, cols },
        })
    }
}

// ---------------- Benches ----------------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const MANY_COLS: usize = 1024;
    const MANY_ROWS: usize = 8192;

    fn bytes_one_series_many_params(rows: usize) -> usize {
        let in_b = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_b = ONE_SERIES_LEN * rows * std::mem::size_of::<f32>() * 2; // K + D
        in_b + out_b + 64 * 1024 * 1024
    }
    fn bytes_many_series() -> usize {
        let n = MANY_COLS * MANY_ROWS;
        let in_b = n * std::mem::size_of::<f32>();
        let out_b = n * std::mem::size_of::<f32>() * 2;
        in_b + out_b + 64 * 1024 * 1024
    }

    struct SrsiBatchState {
        cuda: CudaSrsi,
        prices: Vec<f32>,
        sweep: SrsiBatchRange,
        rows: usize,
    }
    impl CudaBenchState for SrsiBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .srsi_batch_dev(&self.prices, &self.sweep)
                .expect("srsi batch");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaSrsi::new(0).expect("cuda srsi");
        let mut prices = gen_series(ONE_SERIES_LEN);
        
        for i in 0..16 {
            prices[i] = f32::NAN;
        }
        for i in 16..ONE_SERIES_LEN {
            let x = i as f32 * 0.0031;
            prices[i] += 0.001 * x.sin();
        }
        let sweep = SrsiBatchRange {
            rsi_period: (2, 50, 1),
            stoch_period: (2, 32, 1),
            k: (3, 5, 1),
            d: (3, 5, 1),
        };
        let rows = (sweep.rsi_period.1 - sweep.rsi_period.0 + 1)
            * (sweep.stoch_period.1 - sweep.stoch_period.0 + 1)
            * (sweep.k.1 - sweep.k.0 + 1)
            * (sweep.d.1 - sweep.d.0 + 1);
        Box::new(SrsiBatchState { cuda, prices, sweep, rows })
    }

    struct SrsiManyState {
        cuda: CudaSrsi,
        prices_tm: Vec<f32>,
    }
    impl CudaBenchState for SrsiManyState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .srsi_many_series_one_param_time_major_dev(
                    &self.prices_tm,
                    MANY_COLS,
                    MANY_ROWS,
                    &SrsiParams::default(),
                )
                .expect("srsi many");
        }
    }
    fn prep_many_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaSrsi::new(0).expect("cuda srsi");
        let n = MANY_COLS * MANY_ROWS;
        let mut base = gen_series(n);
        let mut prices = vec![f32::NAN; n];
        for s in 0..MANY_COLS {
            for t in s..MANY_ROWS {
                let idx = t * MANY_COLS + s;
                let x = (t as f32) * 0.002 + (s as f32) * 0.01;
                prices[idx] = base[idx] + 0.02 * x.cos();
            }
        }
        Box::new(SrsiManyState { cuda, prices_tm: prices })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let rows = (50 - 2 + 1) * (32 - 2 + 1) * (5 - 3 + 1) * (5 - 3 + 1);
        vec![
            CudaBenchScenario::new(
                "srsi",
                "one_series_many_params",
                "srsi_cuda_batch_dev",
                "1m_param_sweep",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params(rows)),
            CudaBenchScenario::new(
                "srsi",
                "many_series_one_param",
                "srsi_cuda_many_series_one_param_dev",
                "1024x8192",
                prep_many_series,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_many_series()),
        ]
    }
}
