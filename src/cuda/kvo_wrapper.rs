//! CUDA support for Klinger Volume Oscillator (KVO).
//!
//! Pattern: recurrence/IIR.
//! - Batch (one series × many params): precompute VF on host once, then per-row EMA scan on device.
//! - Many-series × one-param (time-major): per-series VF + EMA scan inside the kernel.
//!
//! Semantics: identical to scalar `kvo.rs` implementation.
//! - Warmup: NaN up to `first_valid + 1`; seed EMAs at that index and write 0.0.
//! - f64 accumulations for EMAs and VF intermediates; outputs f32.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
use crate::indicators::kvo::{KvoBatchRange, KvoParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaKvoError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaKvoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaKvoError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaKvoError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaKvoError {}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32, block_y: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaKvoPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaKvoPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

pub struct CudaKvo {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaKvoPolicy,
}

impl CudaKvo {
    pub fn new(device_id: usize) -> Result<Self, CudaKvoError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaKvoError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaKvoError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaKvoError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/kvo_kernel.ptx"));
        // Prefer default JIT optimization level (O4) and target from context.
        let module = Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                // If register pressure becomes a limit, consider:
                // ModuleJitOption::MaxRegisters(128),
            ],
        )
        .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
        .or_else(|_| Module::from_ptx(ptx, &[]))
        .map_err(|e| CudaKvoError::Cuda(e.to_string()))?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaKvoError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaKvoPolicy::default(),
        })
    }

    pub fn set_policy(&mut self, policy: CudaKvoPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaKvoPolicy {
        &self.policy
    }

    // -------------------- Batch: one series × many params --------------------
    pub fn kvo_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        volume: &[f32],
        sweep: &KvoBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<KvoParams>), CudaKvoError> {
        if high.is_empty() || low.is_empty() || close.is_empty() || volume.is_empty() {
            return Err(CudaKvoError::InvalidInput("empty input".into()));
        }
        let len = high.len();
        if low.len() != len || close.len() != len || volume.len() != len {
            return Err(CudaKvoError::InvalidInput(
                "inputs must have equal length".into(),
            ));
        }
        let first = first_valid_ohlcv(high, low, close, volume)
            .ok_or_else(|| CudaKvoError::InvalidInput("all values are NaN".into()))?;
        if len - first < 2 {
            return Err(CudaKvoError::InvalidInput(
                "not enough valid data (need >=2 after first)".into(),
            ));
        }

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaKvoError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let mut shorts = Vec::with_capacity(combos.len());
        let mut longs = Vec::with_capacity(combos.len());
        for c in &combos {
            let s = c.short_period.unwrap_or(0);
            let l = c.long_period.unwrap_or(0);
            if s == 0 || l < s {
                return Err(CudaKvoError::InvalidInput("invalid (short,long)".into()));
            }
            shorts.push(s as i32);
            longs.push(l as i32);
        }

        // Precompute VF once (f32) for device, matching scalar semantics
        let vf = precompute_vf_f32(high, low, close, volume, first);

        // VRAM estimate: vf + shorts + longs + outputs + headroom
        let bytes = len * 4 + combos.len() * 4 * 2 + combos.len() * len * 4 + 64 * 1024 * 1024;
        if let Ok((free, _)) = mem_get_info() {
            if bytes > free {
                return Err(CudaKvoError::InvalidInput(format!(
                    "estimated device memory {:.2} MB exceeds free VRAM",
                    (bytes as f64) / (1024.0 * 1024.0)
                )));
            }
        }

        // H2D
        let d_vf = DeviceBuffer::from_slice(&vf).map_err(|e| CudaKvoError::Cuda(e.to_string()))?;
        let d_shorts =
            DeviceBuffer::from_slice(&shorts).map_err(|e| CudaKvoError::Cuda(e.to_string()))?;
        let d_longs =
            DeviceBuffer::from_slice(&longs).map_err(|e| CudaKvoError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(len * combos.len())
                .map_err(|e| CudaKvoError::Cuda(e.to_string()))?
        };

        self.launch_batch_kernel(
            &d_vf,
            len as i32,
            first as i32,
            &d_shorts,
            &d_longs,
            combos.len() as i32,
            &mut d_out,
        )?;

        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows: combos.len(),
                cols: len,
            },
            combos,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel(
        &self,
        d_vf: &DeviceBuffer<f32>,
        len: i32,
        first_valid: i32,
        d_shorts: &DeviceBuffer<i32>,
        d_longs: &DeviceBuffer<i32>,
        n_combos: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaKvoError> {
        if len <= 0 || n_combos <= 0 {
            return Ok(());
        }
        let func = self
            .module
            .get_function("kvo_batch_f32")
            .map_err(|e| CudaKvoError::Cuda(e.to_string()))?;

        let block_x = match self.policy.batch {
            BatchKernelPolicy::OneD { block_x } if block_x > 0 => block_x,
            _ => 256,
        };
        // 1-D grid over combos (kernel grid-strides over combo index)
        let threads = block_x;
        let blocks = ((n_combos as u32) + threads - 1) / threads;
        let grid: GridSize = (blocks.max(1), 1, 1).into();
        let block: BlockSize = (threads, 1, 1).into();
        unsafe {
            let mut p_vf = d_vf.as_device_ptr().as_raw();
            let mut p_len = len;
            let mut p_first = first_valid;
            let mut p_shorts = d_shorts.as_device_ptr().as_raw();
            let mut p_longs = d_longs.as_device_ptr().as_raw();
            let mut p_n = n_combos;
            let mut p_out = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_vf as *mut _ as *mut c_void,
                &mut p_len as *mut _ as *mut c_void,
                &mut p_first as *mut _ as *mut c_void,
                &mut p_shorts as *mut _ as *mut c_void,
                &mut p_longs as *mut _ as *mut c_void,
                &mut p_n as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaKvoError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    // --------------- Many-series × one-param (time-major) ---------------
    pub fn kvo_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        volume_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &KvoParams,
    ) -> Result<DeviceArrayF32, CudaKvoError> {
        if cols == 0 || rows == 0 {
            return Err(CudaKvoError::InvalidInput("empty matrix".into()));
        }
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaKvoError::InvalidInput("overflow".into()))?;
        if high_tm.len() != elems
            || low_tm.len() != elems
            || close_tm.len() != elems
            || volume_tm.len() != elems
        {
            return Err(CudaKvoError::InvalidInput(
                "inputs must be time-major with equal size".into(),
            ));
        }

        let s = params.short_period.unwrap_or(0);
        let l = params.long_period.unwrap_or(0);
        if s == 0 || l < s {
            return Err(CudaKvoError::InvalidInput("invalid (short,long)".into()));
        }

        // Per-series first-valid detection
        let first_valids =
            first_valids_time_major(high_tm, low_tm, close_tm, volume_tm, cols, rows);

        // VRAM estimate: inputs + first_valids + outputs + headroom
        let bytes = (elems * 4) * 4 + cols * 4 + elems * 4 + 64 * 1024 * 1024;
        if let Ok((free, _)) = mem_get_info() {
            if bytes > free {
                return Err(CudaKvoError::InvalidInput(format!(
                    "estimated device memory {:.2} MB exceeds free VRAM",
                    (bytes as f64) / (1024.0 * 1024.0)
                )));
            }
        }

        // H2D
        let d_high =
            DeviceBuffer::from_slice(high_tm).map_err(|e| CudaKvoError::Cuda(e.to_string()))?;
        let d_low =
            DeviceBuffer::from_slice(low_tm).map_err(|e| CudaKvoError::Cuda(e.to_string()))?;
        let d_close =
            DeviceBuffer::from_slice(close_tm).map_err(|e| CudaKvoError::Cuda(e.to_string()))?;
        let d_vol =
            DeviceBuffer::from_slice(volume_tm).map_err(|e| CudaKvoError::Cuda(e.to_string()))?;
        let d_fv = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaKvoError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(elems).map_err(|e| CudaKvoError::Cuda(e.to_string()))?
        };

        self.launch_many_series_kernel(
            &d_high,
            &d_low,
            &d_close,
            &d_vol,
            &d_fv,
            cols as i32,
            rows as i32,
            s as i32,
            l as i32,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        d_vol: &DeviceBuffer<f32>,
        d_fv: &DeviceBuffer<i32>,
        cols: i32,
        rows: i32,
        short_p: i32,
        long_p: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaKvoError> {
        if cols <= 0 || rows <= 0 {
            return Ok(());
        }
        let func = self
            .module
            .get_function("kvo_many_series_one_param_time_major_f32")
            .map_err(|e| CudaKvoError::Cuda(e.to_string()))?;

        // 1-D over columns; ignore block_y in policy for API compatibility
        let (block_x, _ignore) = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x, block_y: _ } if block_x > 0 => (block_x, 1u32),
            _ => (256, 1u32),
        };
        let threads = block_x;
        let blocks = ((cols as u32) + threads - 1) / threads;
        let grid: GridSize = (blocks.max(1), 1, 1).into();
        let block: BlockSize = (threads, 1, 1).into();

        unsafe {
            let mut p_high = d_high.as_device_ptr().as_raw();
            let mut p_low = d_low.as_device_ptr().as_raw();
            let mut p_close = d_close.as_device_ptr().as_raw();
            let mut p_vol = d_vol.as_device_ptr().as_raw();
            let mut p_fv = d_fv.as_device_ptr().as_raw();
            let mut p_cols = cols;
            let mut p_rows = rows;
            let mut p_sp = short_p;
            let mut p_lp = long_p;
            let mut p_out = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_high as *mut _ as *mut c_void,
                &mut p_low as *mut _ as *mut c_void,
                &mut p_close as *mut _ as *mut c_void,
                &mut p_vol as *mut _ as *mut c_void,
                &mut p_fv as *mut _ as *mut c_void,
                &mut p_cols as *mut _ as *mut c_void,
                &mut p_rows as *mut _ as *mut c_void,
                &mut p_sp as *mut _ as *mut c_void,
                &mut p_lp as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaKvoError::Cuda(e.to_string()))?;
        }
        Ok(())
    }
}

// -------- Helpers --------

fn first_valid_ohlcv(h: &[f32], l: &[f32], c: &[f32], v: &[f32]) -> Option<usize> {
    h.iter()
        .zip(l.iter())
        .zip(c.iter())
        .zip(v.iter())
        .position(|(((hh, ll), cc), vv)| {
            !hh.is_nan() && !ll.is_nan() && !cc.is_nan() && !vv.is_nan()
        })
}

fn precompute_vf_f32(h: &[f32], l: &[f32], c: &[f32], v: &[f32], first: usize) -> Vec<f32> {
    let len = h.len();
    let mut out = vec![f32::NAN; len];
    if len <= first + 1 {
        return out;
    }
    unsafe {
        let hp = h.as_ptr();
        let lp = l.as_ptr();
        let cp = c.as_ptr();
        let vp = v.as_ptr();
        let mut trend: i32 = -1;
        let mut cm: f64 = 0.0;
        let mut prev_hlc =
            (*hp.add(first) as f64) + (*lp.add(first) as f64) + (*cp.add(first) as f64);
        let mut prev_dm = (*hp.add(first) as f64) - (*lp.add(first) as f64);
        let mut i = first + 1;
        while i < len {
            let h = *hp.add(i) as f64;
            let l = *lp.add(i) as f64;
            let c = *cp.add(i) as f64;
            let vol = *vp.add(i) as f64;
            let hlc = h + l + c;
            let dm = h - l;
            if hlc > prev_hlc && trend != 1 {
                trend = 1;
                cm = prev_dm;
            } else if hlc < prev_hlc && trend != 0 {
                trend = 0;
                cm = prev_dm;
            }
            cm += dm;
            let temp = (((dm / cm) * 2.0) - 1.0).abs();
            let sign = if trend == 1 { 1.0 } else { -1.0 };
            let vf = vol * temp * 100.0 * sign;
            out[i] = vf as f32;
            prev_hlc = hlc;
            prev_dm = dm;
            i += 1;
        }
    }
    out
}

fn first_valids_time_major(
    h_tm: &[f32],
    l_tm: &[f32],
    c_tm: &[f32],
    v_tm: &[f32],
    cols: usize,
    rows: usize,
) -> Vec<i32> {
    let mut fv = vec![-1i32; cols];
    for s in 0..cols {
        for t in 0..rows {
            let idx = t * cols + s;
            let hh = h_tm[idx];
            let ll = l_tm[idx];
            let cc = c_tm[idx];
            let vv = v_tm[idx];
            if !hh.is_nan() && !ll.is_nan() && !cc.is_nan() && !vv.is_nan() {
                fv[s] = t as i32;
                break;
            }
        }
    }
    fv
}

fn expand_grid(r: &KvoBatchRange) -> Vec<KvoParams> {
    fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        if start > end {
            return Vec::new();
        }
        (start..=end).step_by(step).collect()
    }
    let shorts = axis(r.short_period);
    let longs = axis(r.long_period);
    let mut out = Vec::with_capacity(shorts.len() * longs.len());
    for &s in &shorts {
        for &l in &longs {
            if s >= 1 && l >= s {
                out.push(KvoParams {
                    short_period: Some(s),
                    long_period: Some(l),
                });
            }
        }
    }
    out
}

#[inline(always)]
fn grid_y_chunks(n: usize) -> impl Iterator<Item = (usize, usize)> {
    struct YChunks {
        n: usize,
        launched: usize,
    }
    impl Iterator for YChunks {
        type Item = (usize, usize);
        fn next(&mut self) -> Option<Self::Item> {
            const MAX: usize = 65_535;
            if self.launched >= self.n {
                return None;
            }
            let start = self.launched;
            let len = (self.n - self.launched).min(MAX);
            self.launched += len;
            Some((start, len))
        }
    }
    YChunks { n, launched: 0 }
}

// ---------------- Benches ----------------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{
        gen_series, gen_time_major_prices, gen_time_major_volumes, gen_volume,
    };
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const SHORT_RANGE: (usize, usize, usize) = (2, 16, 1);
    const LONG_RANGE: (usize, usize, usize) = (18, 50, 2); // ~255 combos
    const MANY_SERIES_COLS: usize = 256;
    const MANY_SERIES_ROWS: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let combos = ((SHORT_RANGE.1 - SHORT_RANGE.0) / SHORT_RANGE.2 + 1)
            * ((LONG_RANGE.1 - LONG_RANGE.0) / LONG_RANGE.2 + 1);
        let in_bytes = ONE_SERIES_LEN * 4 * 4; // high/low/close/volume f32 (host used only for VF; conservative)
        let vf_bytes = ONE_SERIES_LEN * 4; // f32 VF
        let out_bytes = ONE_SERIES_LEN * combos * 4;
        in_bytes + vf_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_ROWS;
        let in_bytes = elems * 4 * 4; // hlcv
        let out_bytes = elems * 4;
        let fv_bytes = MANY_SERIES_COLS * 4;
        in_bytes + out_bytes + fv_bytes + 64 * 1024 * 1024
    }

    struct KvoBatchState {
        cuda: CudaKvo,
        h: Vec<f32>,
        l: Vec<f32>,
        c: Vec<f32>,
        v: Vec<f32>,
        sweep: KvoBatchRange,
    }
    impl CudaBenchState for KvoBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .kvo_batch_dev(&self.h, &self.l, &self.c, &self.v, &self.sweep)
                .expect("kvo batch");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaKvo::new(0).expect("cuda kvo");
        let price = gen_series(ONE_SERIES_LEN);
        let vol = gen_volume(ONE_SERIES_LEN);
        // Build simple OHLC around price
        let mut h = price.clone();
        let mut l = price.clone();
        let mut c = price.clone();
        for i in 2..ONE_SERIES_LEN {
            let base = price[i];
            h[i] = base + 0.1f32 * (i as f32 * 0.001).sin().abs();
            l[i] = base - 0.1f32 * (i as f32 * 0.001).cos().abs();
            c[i] = base + 0.05f32 * (i as f32 * 0.0013).sin();
        }
        let sweep = KvoBatchRange {
            short_period: SHORT_RANGE,
            long_period: LONG_RANGE,
        };
        Box::new(KvoBatchState {
            cuda,
            h,
            l,
            c,
            v: vol,
            sweep,
        })
    }

    struct KvoManyState {
        cuda: CudaKvo,
        h_tm: Vec<f32>,
        l_tm: Vec<f32>,
        c_tm: Vec<f32>,
        v_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        params: KvoParams,
    }
    impl CudaBenchState for KvoManyState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .kvo_many_series_one_param_time_major_dev(
                    &self.h_tm,
                    &self.l_tm,
                    &self.c_tm,
                    &self.v_tm,
                    self.cols,
                    self.rows,
                    &self.params,
                )
                .expect("kvo many-series");
        }
    }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaKvo::new(0).expect("cuda kvo");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_ROWS;
        let price_tm = gen_time_major_prices(cols, rows);
        let vol_tm = gen_time_major_volumes(cols, rows);
        // derive simple OHLC from price
        let mut h_tm = price_tm.clone();
        let mut l_tm = price_tm.clone();
        let mut c_tm = price_tm.clone();
        for s in 0..cols {
            for t in s..rows {
                let idx = t * cols + s;
                let base = price_tm[idx];
                h_tm[idx] = base + 0.1f32 * (t as f32 * 0.0011).sin().abs();
                l_tm[idx] = base - 0.1f32 * (t as f32 * 0.0012).cos().abs();
                c_tm[idx] = base + 0.03f32 * (t as f32 * 0.0015).sin();
            }
        }
        let params = KvoParams {
            short_period: Some(6),
            long_period: Some(20),
        };
        Box::new(KvoManyState {
            cuda,
            h_tm,
            l_tm,
            c_tm,
            v_tm: vol_tm,
            cols,
            rows,
            params,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "kvo",
                "one_series_many_params",
                "kvo_batch",
                "kvo_batch/rowsxcols",
                prep_one_series_many_params,
            )
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "kvo",
                "many_series_one_param",
                "kvo_many_series",
                "kvo_many/colsxrows",
                prep_many_series_one_param,
            )
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
