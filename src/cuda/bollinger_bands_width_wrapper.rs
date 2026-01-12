#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::bollinger_bands_width::BollingerBandsWidthBatchRange;
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer, DeviceCopy, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Float2 {
    hi: f32,
    lo: f32,
}

unsafe impl DeviceCopy for Float2 {}

#[inline]
fn ff_two_sum(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);

    let t = s + e;
    let e2 = e - (t - s);
    (t, e2)
}
#[inline]
fn ff_add(a: Float2, b: Float2) -> Float2 {
    let (s, mut e) = ff_two_sum(a.hi, b.hi);
    e += a.lo + b.lo;
    let t = s + e;
    let e2 = e - (t - s);
    Float2 { hi: t, lo: e2 }
}
#[inline]
fn ff_prod_fma(x: f32, y: f32) -> Float2 {
    let p = x * y;
    let e = f32::mul_add(x, y, -p);
    let t = p + e;
    let e2 = e - (t - p);
    Float2 { hi: t, lo: e2 }
}

#[derive(Debug, Error)]
pub enum CudaBbwError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error(
        "Out of memory: required={required} bytes, free={free} bytes, headroom={headroom} bytes"
    )]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("Missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("Launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
    #[error("Device mismatch: buffer device {buf}, current {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("Not implemented")]
    NotImplemented,
}

#[derive(Clone, Debug)]
struct BbwCombo {
    period: usize,
    u_plus_d: f32,
}

#[derive(Clone, Debug)]
struct Grouped {
    unique_periods: Vec<i32>,
    offsets: Vec<i32>,
    uplusd_sorted: Vec<f32>,
    combo_index: Vec<i32>,
}

fn group_by_period(combos: &[BbwCombo]) -> Grouped {
    let mut pairs: Vec<(i32, f32, i32)> = combos
        .iter()
        .enumerate()
        .map(|(i, c)| (c.period as i32, c.u_plus_d, i as i32))
        .collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));

    let mut unique = Vec::new();
    let mut offsets = Vec::with_capacity(pairs.len() + 1);
    offsets.push(0);
    let mut uks = Vec::with_capacity(pairs.len());
    let mut rows = Vec::with_capacity(pairs.len());

    let mut count = 0usize;
    let mut cur: Option<i32> = None;
    for (p, k, r) in pairs.into_iter() {
        if cur != Some(p) {
            if cur.is_some() {
                offsets.push(count as i32);
            }
            unique.push(p);
            cur = Some(p);
        }
        uks.push(k);
        rows.push(r);
        count += 1;
    }
    offsets.push(count as i32);

    Grouped {
        unique_periods: unique,
        offsets,
        uplusd_sorted: uks,
        combo_index: rows,
    }
}

fn should_use_grouped(n_combos: usize, unique_periods: usize) -> bool {
    unique_periods < n_combos && (n_combos as f64 / unique_periods as f64) >= 2.0
}

pub struct CudaBbw {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,

    batch_policy: BatchKernelPolicy,
    many_policy: ManySeriesKernelPolicy,
    debug_logged: std::sync::atomic::AtomicBool,
}

impl CudaBbw {
    pub fn new(device_id: usize) -> Result<Self, CudaBbwError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(
            env!("OUT_DIR"),
            "/bollinger_bands_width_kernel.ptx"
        ));
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
            _context: context,
            device_id: device_id as u32,
            batch_policy: BatchKernelPolicy::Auto,
            many_policy: ManySeriesKernelPolicy::Auto,
            debug_logged: std::sync::atomic::AtomicBool::new(false),
        })
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> {
        self._context.clone()
    }
    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    #[inline]
    fn will_fit(&self, required: usize, headroom: usize) -> Result<(), CudaBbwError> {
        match mem_get_info() {
            Ok((free, _)) => {
                if required.saturating_add(headroom) > free {
                    return Err(CudaBbwError::OutOfMemory {
                        required,
                        free,
                        headroom,
                    });
                }
                Ok(())
            }
            Err(e) => Err(CudaBbwError::Cuda(e)),
        }
    }

    pub fn set_policies(&mut self, batch: BatchKernelPolicy, many: ManySeriesKernelPolicy) {
        self.batch_policy = batch;
        self.many_policy = many;
    }

    pub fn bbw_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &BollingerBandsWidthBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<(usize, f32)>), CudaBbwError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let arr = self.run_batch_kernel(data_f32, &combos, first_valid)?;
        self.stream.synchronize()?;
        let meta = combos.iter().map(|c| (c.period, c.u_plus_d)).collect();
        Ok((arr, meta))
    }

    pub fn bbw_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &BollingerBandsWidthBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<(usize, f32)>), CudaBbwError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos
            .len()
            .checked_mul(len)
            .ok_or_else(|| CudaBbwError::InvalidInput("output too large".into()))?;
        if out.len() != expected {
            return Err(CudaBbwError::InvalidInput(format!(
                "output slice length mismatch (expected {}, got {})",
                expected,
                out.len()
            )));
        }
        let dev = self.run_batch_kernel(data_f32, &combos, first_valid)?;
        dev.buf.copy_to(out).map_err(CudaBbwError::Cuda)?;
        let meta = combos.iter().map(|c| (c.period, c.u_plus_d)).collect();
        Ok((combos.len(), len, meta))
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &BollingerBandsWidthBatchRange,
    ) -> Result<(Vec<BbwCombo>, usize, usize), CudaBbwError> {
        if data_f32.is_empty() {
            return Err(CudaBbwError::InvalidInput("empty data".into()));
        }
        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|&v| !v.is_nan())
            .ok_or_else(|| CudaBbwError::InvalidInput("all values are NaN".into()))?;

        let mut periods = Vec::new();
        let (ps, pe, pst) = sweep.period;
        if pst == 0 || ps == pe {
            periods.push(ps);
        } else if ps < pe {
            let mut p = ps;
            while p <= pe {
                periods.push(p);
                match p.checked_add(pst) {
                    Some(n) => p = n,
                    None => break,
                }
            }
        } else {
            let mut p = ps as i64;
            let step_i = pst as i64;
            while p >= pe as i64 {
                periods.push(p as usize);
                p -= step_i;
            }
        }
        let mut devups = Vec::new();
        let (us, ue, ust) = sweep.devup;
        if ust.abs() < 1e-12 || (us - ue).abs() < 1e-12 {
            devups.push(us);
        } else if ust > 0.0 && us <= ue {
            let mut u = us;
            while u <= ue + 1e-12 {
                devups.push(u);
                u += ust;
            }
        } else if ust < 0.0 && us >= ue {
            let mut u = us;
            while u >= ue - 1e-12 {
                devups.push(u);
                u += ust;
            }
        }
        let mut devdns = Vec::new();
        let (ds, de, dst) = sweep.devdn;
        if dst.abs() < 1e-12 || (ds - de).abs() < 1e-12 {
            devdns.push(ds);
        } else if dst > 0.0 && ds <= de {
            let mut d = ds;
            while d <= de + 1e-12 {
                devdns.push(d);
                d += dst;
            }
        } else if dst < 0.0 && ds >= de {
            let mut d = ds;
            while d >= de - 1e-12 {
                devdns.push(d);
                d += dst;
            }
        }

        if periods.is_empty() || devups.is_empty() || devdns.is_empty() {
            return Err(CudaBbwError::InvalidInput(
                "invalid range (empty expansion)".into(),
            ));
        }

        let cap = periods
            .len()
            .checked_mul(devups.len())
            .and_then(|v| v.checked_mul(devdns.len()))
            .ok_or_else(|| CudaBbwError::InvalidInput("range too large".into()))?;
        let mut combos = Vec::with_capacity(cap);
        let mut max_period = 0usize;
        for &p in &periods {
            for &u in &devups {
                for &d in &devdns {
                    combos.push(BbwCombo {
                        period: p,
                        u_plus_d: (u + d) as f32,
                    });
                }
            }
            max_period = max_period.max(p);
        }
        if len - first_valid < max_period {
            return Err(CudaBbwError::InvalidInput(format!(
                "not enough valid data (needed >= {}, valid = {})",
                max_period,
                len - first_valid
            )));
        }
        Ok((combos, first_valid, len))
    }

    fn build_prefixes(data: &[f32]) -> (Vec<Float2>, Vec<Float2>, Vec<i32>) {
        let len = data.len();
        let mut ps = vec![Float2::default(); len + 1];
        let mut ps2 = vec![Float2::default(); len + 1];
        let mut pn = vec![0i32; len + 1];
        let mut acc = Float2::default();
        let mut acc2 = Float2::default();
        let mut acc_n = 0i32;
        for i in 0..len {
            let v = data[i];
            if v.is_nan() {
                acc_n += 1;
            } else {
                acc = ff_add(acc, Float2 { hi: v, lo: 0.0 });
                acc2 = ff_add(acc2, ff_prod_fma(v, v));
            }
            ps[i + 1] = acc;
            ps2[i + 1] = acc2;
            pn[i + 1] = acc_n;
        }
        (ps, ps2, pn)
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[BbwCombo],
        first_valid: usize,
    ) -> Result<DeviceArrayF32, CudaBbwError> {
        let len = data_f32.len();

        let (ps, ps2, pn) = Self::build_prefixes(data_f32);
        let sz_f2 = std::mem::size_of::<Float2>();
        let sz_i32 = std::mem::size_of::<i32>();
        let in_bytes_ps = ps
            .len()
            .checked_mul(sz_f2)
            .ok_or_else(|| CudaBbwError::InvalidInput("size overflow".into()))?;
        let in_bytes_ps2 = ps2
            .len()
            .checked_mul(sz_f2)
            .ok_or_else(|| CudaBbwError::InvalidInput("size overflow".into()))?;
        let in_bytes_pn = pn
            .len()
            .checked_mul(sz_i32)
            .ok_or_else(|| CudaBbwError::InvalidInput("size overflow".into()))?;
        let in_bytes = in_bytes_ps
            .checked_add(in_bytes_ps2)
            .and_then(|v| v.checked_add(in_bytes_pn))
            .ok_or_else(|| CudaBbwError::InvalidInput("size overflow".into()))?;

        let out_elems = len
            .checked_mul(combos.len())
            .ok_or_else(|| CudaBbwError::InvalidInput("output too large".into()))?;
        let out_bytes = out_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaBbwError::InvalidInput("output too large".into()))?;
        let required = in_bytes
            .checked_add(out_bytes)
            .ok_or_else(|| CudaBbwError::InvalidInput("size overflow".into()))?;
        let headroom = 64usize * 1024 * 1024;
        self.will_fit(required, headroom)?;

        let d_ps = DeviceBuffer::from_slice(&ps)?;
        let d_ps2 = DeviceBuffer::from_slice(&ps2)?;
        let d_pn = DeviceBuffer::from_slice(&pn)?;

        let periods: Vec<i32> = combos.iter().map(|c| c.period as i32).collect();
        let uplusd: Vec<f32> = combos.iter().map(|c| c.u_plus_d).collect();
        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let d_uplusd = DeviceBuffer::from_slice(&uplusd)?;

        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(out_elems) }?;

        if !self.debug_logged.load(std::sync::atomic::Ordering::Relaxed)
            && std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1")
        {
            eprintln!(
                "[bbw] policy={:?}/{:?} len={} rows={} (float-float)",
                self.batch_policy,
                self.many_policy,
                len,
                combos.len()
            );
            self.debug_logged
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }

        let grouped = {
            let g = group_by_period(combos);
            if should_use_grouped(combos.len(), g.unique_periods.len()) {
                Some(g)
            } else {
                None
            }
        };

        if let Some(g) = grouped {
            let d_up = DeviceBuffer::from_slice(&g.unique_periods)?;
            let d_offs = DeviceBuffer::from_slice(&g.offsets)?;
            let d_uks = DeviceBuffer::from_slice(&g.uplusd_sorted)?;
            let d_rows = DeviceBuffer::from_slice(&g.combo_index)?;

            self.launch_grouped_batch_kernel(
                &d_ps,
                &d_ps2,
                &d_pn,
                len,
                first_valid,
                &d_up,
                &d_offs,
                &d_uks,
                &d_rows,
                g.unique_periods.len(),
                d_out.as_device_ptr(),
            )?;
        } else {
            let use_streaming = combos.len() <= 64;
            if use_streaming {
                let d_data = DeviceBuffer::from_slice(data_f32)?;
                self.launch_batch_kernel_streaming(
                    &d_data,
                    len,
                    first_valid,
                    d_periods.as_device_ptr(),
                    d_uplusd.as_device_ptr(),
                    combos.len(),
                    d_out.as_device_ptr(),
                )?;
            } else {
                self.launch_batch_kernel_ptrs(
                    &d_ps,
                    &d_ps2,
                    &d_pn,
                    d_periods.as_device_ptr(),
                    d_uplusd.as_device_ptr(),
                    len,
                    first_valid,
                    combos.len(),
                    d_out.as_device_ptr(),
                )?;
            }
        }

        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: combos.len(),
            cols: len,
        })
    }

    fn launch_batch_kernel_streaming(
        &self,
        d_data: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        periods_ptr: cust::memory::DevicePointer<i32>,
        uplusd_ptr: cust::memory::DevicePointer<f32>,
        n_combos: usize,
        out_ptr: cust::memory::DevicePointer<f32>,
    ) -> Result<(), CudaBbwError> {
        let func = self
            .module
            .get_function("bbw_sma_streaming_f64")
            .map_err(|_| CudaBbwError::MissingKernelSymbol {
                name: "bbw_sma_streaming_f64",
            })?;
        if len > i32::MAX as usize || n_combos > i32::MAX as usize {
            return Err(CudaBbwError::InvalidInput("input too large".into()));
        }
        let block: BlockSize = (1, 1, 1).into();
        let grid: GridSize = (1, n_combos as u32, 1).into();
        unsafe {
            let mut data_ptr = d_data.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut first_i = first_valid as i32;
            let mut per_ptr = periods_ptr.as_raw();
            let mut up_ptr = uplusd_ptr.as_raw();
            let mut n_i = n_combos as i32;
            let mut out = out_ptr.as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut data_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut per_ptr as *mut _ as *mut c_void,
                &mut up_ptr as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut out as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaBbwError::Cuda)?;
        }
        Ok(())
    }

    fn launch_batch_kernel_ptrs(
        &self,
        d_ps: &DeviceBuffer<Float2>,
        d_ps2: &DeviceBuffer<Float2>,
        d_pn: &DeviceBuffer<i32>,
        periods_ptr: cust::memory::DevicePointer<i32>,
        uplusd_ptr: cust::memory::DevicePointer<f32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        out_ptr: cust::memory::DevicePointer<f32>,
    ) -> Result<(), CudaBbwError> {
        let func = self
            .module
            .get_function("bbw_sma_prefix_ff_f32")
            .map_err(|_| CudaBbwError::MissingKernelSymbol {
                name: "bbw_sma_prefix_ff_f32",
            })?;

        if len > i32::MAX as usize || n_combos > i32::MAX as usize {
            return Err(CudaBbwError::InvalidInput(
                "input too large for kernel argument width".into(),
            ));
        }

        let block_x: u32 = match self.batch_policy {
            BatchKernelPolicy::Plain { block_x } if block_x > 0 => block_x,
            _ => 256,
        };
        let grid_x = ((len as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), n_combos as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut ps_ptr = d_ps.as_device_ptr().as_raw();
            let mut ps2_ptr = d_ps2.as_device_ptr().as_raw();
            let mut pn_ptr = d_pn.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut first_valid_i = first_valid as i32;
            let mut periods_ptr = periods_ptr.as_raw();
            let mut uplusd_ptr = uplusd_ptr.as_raw();
            let mut combos_i = n_combos as i32;
            let mut out_ptr = out_ptr.as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut ps_ptr as *mut _ as *mut c_void,
                &mut ps2_ptr as *mut _ as *mut c_void,
                &mut pn_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut uplusd_ptr as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaBbwError::Cuda)?;
        }
        Ok(())
    }

    fn launch_grouped_batch_kernel(
        &self,
        d_ps: &DeviceBuffer<Float2>,
        d_ps2: &DeviceBuffer<Float2>,
        d_pn: &DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        d_unique: &DeviceBuffer<i32>,
        d_offsets: &DeviceBuffer<i32>,
        d_uplusd_sorted: &DeviceBuffer<f32>,
        d_combo_index: &DeviceBuffer<i32>,
        num_unique: usize,
        out_ptr: cust::memory::DevicePointer<f32>,
    ) -> Result<(), CudaBbwError> {
        let func = self
            .module
            .get_function("bbw_sma_prefix_grouped_ff_f32")
            .map_err(|_| CudaBbwError::MissingKernelSymbol {
                name: "bbw_sma_prefix_grouped_ff_f32",
            })?;

        if len > i32::MAX as usize || num_unique > i32::MAX as usize {
            return Err(CudaBbwError::InvalidInput(
                "arguments exceed kernel limits".into(),
            ));
        }
        let block_x: u32 = match self.batch_policy {
            BatchKernelPolicy::Plain { block_x } if block_x > 0 => block_x,
            _ => 256,
        };
        let grid_x = ((len as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), num_unique as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut ps_ptr = d_ps.as_device_ptr().as_raw();
            let mut ps2_ptr = d_ps2.as_device_ptr().as_raw();
            let mut pn_ptr = d_pn.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut first_valid_i = first_valid as i32;
            let mut up_ptr = d_unique.as_device_ptr().as_raw();
            let mut off_ptr = d_offsets.as_device_ptr().as_raw();
            let mut uks_ptr = d_uplusd_sorted.as_device_ptr().as_raw();
            let mut rows_ptr = d_combo_index.as_device_ptr().as_raw();
            let mut num_unique_i = num_unique as i32;
            let mut out_ptr = out_ptr.as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut ps_ptr as *mut _ as *mut c_void,
                &mut ps2_ptr as *mut _ as *mut c_void,
                &mut pn_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut up_ptr as *mut _ as *mut c_void,
                &mut off_ptr as *mut _ as *mut c_void,
                &mut uks_ptr as *mut _ as *mut c_void,
                &mut rows_ptr as *mut _ as *mut c_void,
                &mut num_unique_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaBbwError::Cuda)?;
        }
        Ok(())
    }

    pub fn bbw_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        devup: f32,
        devdn: f32,
    ) -> Result<DeviceArrayF32, CudaBbwError> {
        let prep = Self::prepare_many_series_inputs(data_tm_f32, cols, rows, period)?;
        self.run_many_series_kernel(&prep, cols, rows, period, devup + devdn)
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<ManySeriesPrepared, CudaBbwError> {
        if cols == 0 || rows == 0 {
            return Err(CudaBbwError::InvalidInput(
                "matrix dims must be positive".into(),
            ));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaBbwError::InvalidInput("matrix dims overflow".into()))?;
        if data_tm_f32.len() != expected {
            return Err(CudaBbwError::InvalidInput("matrix shape mismatch".into()));
        }
        if period == 0 {
            return Err(CudaBbwError::InvalidInput("period must be > 0".into()));
        }

        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let idx = t * cols + s;
                let v = data_tm_f32[idx];
                if !v.is_nan() {
                    fv = Some(t);
                    break;
                }
            }
            let fv =
                fv.ok_or_else(|| CudaBbwError::InvalidInput(format!("series {} has all NaN", s)))?;
            if rows - fv < period {
                return Err(CudaBbwError::InvalidInput(format!(
                    "series {} lacks data: needed >= {}, valid = {}",
                    s,
                    period,
                    rows - fv
                )));
            }
            first_valids[s] = fv as i32;
        }

        let (ps_tm, ps2_tm, pn_tm) =
            compute_prefix_sums_time_major_ff(data_tm_f32, cols, rows, &first_valids);
        Ok(ManySeriesPrepared {
            first_valids,
            ps_tm,
            ps2_tm,
            pn_tm,
            data_tm: data_tm_f32.to_vec(),
        })
    }

    fn run_many_series_kernel(
        &self,
        prep: &ManySeriesPrepared,
        cols: usize,
        rows: usize,
        period: usize,
        uplusd: f32,
    ) -> Result<DeviceArrayF32, CudaBbwError> {
        let total = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaBbwError::InvalidInput("matrix dims overflow".into()))?;
        let sz_f32 = std::mem::size_of::<f32>();
        let sz_f2 = std::mem::size_of::<Float2>();
        let sz_i32 = std::mem::size_of::<i32>();
        let headroom = 64usize * 1024 * 1024;

        if cols <= 64 {
            let out_bytes = total
                .checked_mul(sz_f32)
                .ok_or_else(|| CudaBbwError::InvalidInput("size overflow".into()))?;
            let data_bytes = prep
                .data_tm
                .len()
                .checked_mul(sz_f32)
                .ok_or_else(|| CudaBbwError::InvalidInput("size overflow".into()))?;
            let first_bytes = prep
                .first_valids
                .len()
                .checked_mul(sz_i32)
                .ok_or_else(|| CudaBbwError::InvalidInput("size overflow".into()))?;
            let required = out_bytes
                .checked_add(data_bytes)
                .and_then(|v| v.checked_add(first_bytes))
                .ok_or_else(|| CudaBbwError::InvalidInput("size overflow".into()))?;
            self.will_fit(required, headroom)?;

            let mut d_out_tm = unsafe { DeviceBuffer::<f32>::uninitialized(total) }?;
            let d_data_tm = DeviceBuffer::from_slice(&prep.data_tm)?;
            let d_first = DeviceBuffer::from_slice(&prep.first_valids)?;
            self.launch_many_series_kernel_streaming(
                &d_data_tm,
                period,
                cols,
                rows,
                &d_first,
                uplusd,
                &mut d_out_tm,
            )?;

            self.stream.synchronize()?;

            return Ok(DeviceArrayF32 {
                buf: d_out_tm,
                rows,
                cols,
            });
        } else {
            let out_bytes = total
                .checked_mul(sz_f32)
                .ok_or_else(|| CudaBbwError::InvalidInput("size overflow".into()))?;
            let ps_bytes = prep
                .ps_tm
                .len()
                .checked_mul(sz_f2)
                .ok_or_else(|| CudaBbwError::InvalidInput("size overflow".into()))?;
            let ps2_bytes = prep
                .ps2_tm
                .len()
                .checked_mul(sz_f2)
                .ok_or_else(|| CudaBbwError::InvalidInput("size overflow".into()))?;
            let pn_bytes = prep
                .pn_tm
                .len()
                .checked_mul(sz_i32)
                .ok_or_else(|| CudaBbwError::InvalidInput("size overflow".into()))?;
            let first_bytes = prep
                .first_valids
                .len()
                .checked_mul(sz_i32)
                .ok_or_else(|| CudaBbwError::InvalidInput("size overflow".into()))?;
            let required = out_bytes
                .checked_add(ps_bytes)
                .and_then(|v| v.checked_add(ps2_bytes))
                .and_then(|v| v.checked_add(pn_bytes))
                .and_then(|v| v.checked_add(first_bytes))
                .ok_or_else(|| CudaBbwError::InvalidInput("size overflow".into()))?;
            self.will_fit(required, headroom)?;

            let mut d_out_tm = unsafe { DeviceBuffer::<f32>::uninitialized(total) }?;
            let d_ps_tm = DeviceBuffer::from_slice(&prep.ps_tm)?;
            let d_ps2_tm = DeviceBuffer::from_slice(&prep.ps2_tm)?;
            let d_pn_tm = DeviceBuffer::from_slice(&prep.pn_tm)?;
            let d_first = DeviceBuffer::from_slice(&prep.first_valids)?;
            self.launch_many_series_kernel(
                &d_ps_tm,
                &d_ps2_tm,
                &d_pn_tm,
                period,
                cols,
                rows,
                &d_first,
                uplusd,
                &mut d_out_tm,
            )?;

            self.stream.synchronize()?;

            Ok(DeviceArrayF32 {
                buf: d_out_tm,
                rows,
                cols,
            })
        }
    }

    fn launch_many_series_kernel(
        &self,
        d_ps_tm: &DeviceBuffer<Float2>,
        d_ps2_tm: &DeviceBuffer<Float2>,
        d_pn_tm: &DeviceBuffer<i32>,
        period: usize,
        cols: usize,
        rows: usize,
        d_first: &DeviceBuffer<i32>,
        uplusd: f32,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaBbwError> {
        let func = self
            .module
            .get_function("bbw_multi_series_one_param_tm_ff_f32")
            .map_err(|_| CudaBbwError::MissingKernelSymbol {
                name: "bbw_multi_series_one_param_tm_ff_f32",
            })?;
        if period > i32::MAX as usize || cols > i32::MAX as usize || rows > i32::MAX as usize {
            return Err(CudaBbwError::InvalidInput(
                "arguments exceed kernel limits".into(),
            ));
        }
        let block_x: u32 = match self.many_policy {
            ManySeriesKernelPolicy::OneD { block_x } if block_x > 0 => block_x,
            _ => 256,
        };
        let grid_x = ((rows as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), cols as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut ps_ptr = d_ps_tm.as_device_ptr().as_raw();
            let mut ps2_ptr = d_ps2_tm.as_device_ptr().as_raw();
            let mut pn_ptr = d_pn_tm.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut first_ptr = d_first.as_device_ptr().as_raw();
            let mut u_k = uplusd as f32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut ps_ptr as *mut _ as *mut c_void,
                &mut ps2_ptr as *mut _ as *mut c_void,
                &mut pn_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut u_k as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaBbwError::Cuda)?;
        }
        Ok(())
    }

    fn launch_many_series_kernel_streaming(
        &self,
        d_data_tm: &DeviceBuffer<f32>,
        period: usize,
        cols: usize,
        rows: usize,
        d_first: &DeviceBuffer<i32>,
        uplusd: f32,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaBbwError> {
        let func = self
            .module
            .get_function("bbw_multi_series_one_param_tm_streaming_f64")
            .map_err(|_| CudaBbwError::MissingKernelSymbol {
                name: "bbw_multi_series_one_param_tm_streaming_f64",
            })?;
        if period > i32::MAX as usize || cols > i32::MAX as usize || rows > i32::MAX as usize {
            return Err(CudaBbwError::InvalidInput(
                "arguments exceed kernel limits".into(),
            ));
        }
        let block: BlockSize = (1, 1, 1).into();
        let grid: GridSize = (1, cols as u32, 1).into();
        unsafe {
            let mut data_ptr = d_data_tm.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut first_ptr = d_first.as_device_ptr().as_raw();
            let mut u = uplusd as f32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut data_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut u as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaBbwError::Cuda)?;
        }
        Ok(())
    }
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

struct ManySeriesPrepared {
    first_valids: Vec<i32>,
    ps_tm: Vec<Float2>,
    ps2_tm: Vec<Float2>,
    pn_tm: Vec<i32>,
    data_tm: Vec<f32>,
}

fn compute_prefix_sums_time_major_ff(
    data_tm: &[f32],
    cols: usize,
    rows: usize,
    first_valids: &[i32],
) -> (Vec<Float2>, Vec<Float2>, Vec<i32>) {
    let total = data_tm.len();
    let mut ps = vec![Float2::default(); total + 1];
    let mut ps2 = vec![Float2::default(); total + 1];
    let mut pn = vec![0i32; total + 1];

    for s in 0..cols {
        let fv = first_valids[s].max(0) as usize;
        let mut acc = Float2::default();
        let mut acc2 = Float2::default();
        let mut acc_n = 0i32;
        for t in 0..rows {
            let idx = t * cols + s;
            if t >= fv {
                let v = data_tm[idx];
                if v.is_nan() {
                    acc_n += 1;
                } else {
                    acc = ff_add(acc, Float2 { hi: v, lo: 0.0 });
                    acc2 = ff_add(acc2, ff_prod_fma(v, v));
                }
            }
            let w = idx + 1;
            ps[w] = acc;
            ps2[w] = acc2;
            pn[w] = acc_n;
        }
    }
    (ps, ps2, pn)
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::bollinger_bands_width::BollingerBandsWidthBatchRange;

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;

    fn bytes_one_series_many_params() -> usize {
        let prefix =
            (ONE_SERIES_LEN + 1) * (2 * std::mem::size_of::<Float2>() + std::mem::size_of::<i32>());
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        prefix + out_bytes + 64 * 1024 * 1024
    }

    struct BbwBatchState {
        cuda: CudaBbw,
        d_ps: DeviceBuffer<Float2>,
        d_ps2: DeviceBuffer<Float2>,
        d_pn: DeviceBuffer<i32>,
        d_periods: DeviceBuffer<i32>,
        d_uplusd: DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for BbwBatchState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel_ptrs(
                    &self.d_ps,
                    &self.d_ps2,
                    &self.d_pn,
                    self.d_periods.as_device_ptr(),
                    self.d_uplusd.as_device_ptr(),
                    self.len,
                    self.first_valid,
                    self.n_combos,
                    self.d_out.as_device_ptr(),
                )
                .expect("bbw launch");
            self.cuda.stream.synchronize().expect("bbw sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaBbw::new(0).expect("cuda bbw");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = BollingerBandsWidthBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
            devup: (2.0, 2.0, 0.0),
            devdn: (2.0, 2.0, 0.0),
        };

        let (combos, first_valid, len) =
            CudaBbw::prepare_batch_inputs(&price, &sweep).expect("prepare batch");
        let (ps, ps2, pn) = CudaBbw::build_prefixes(&price);

        let periods: Vec<i32> = combos.iter().map(|c| c.period as i32).collect();
        let uplusd: Vec<f32> = combos.iter().map(|c| c.u_plus_d).collect();

        let d_ps = DeviceBuffer::from_slice(&ps).expect("ps H2D");
        let d_ps2 = DeviceBuffer::from_slice(&ps2).expect("ps2 H2D");
        let d_pn = DeviceBuffer::from_slice(&pn).expect("pn H2D");
        let d_periods = DeviceBuffer::from_slice(&periods).expect("periods H2D");
        let d_uplusd = DeviceBuffer::from_slice(&uplusd).expect("uplusd H2D");

        let elems = combos.len() * len;
        let d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }.expect("out");

        Box::new(BbwBatchState {
            cuda,
            d_ps,
            d_ps2,
            d_pn,
            d_periods,
            d_uplusd,
            len,
            first_valid,
            n_combos: combos.len(),
            d_out,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "bollinger_bands_width",
            "one_series_many_params",
            "bollinger_bands_width_cuda_batch_dev",
            "1m_x_250",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
