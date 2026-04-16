#![cfg(feature = "cuda")]

use crate::indicators::vwap_deviation_oscillator::{
    expand_grid, VwapDeviationMode, VwapDeviationOscillatorBatchRange,
    VwapDeviationOscillatorParams, VwapDeviationSessionMode,
};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use thiserror::Error;

const VWAP_DEVIATION_OSCILLATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_ROLLING_PERIOD: usize = 20;
const DEFAULT_ROLLING_DAYS: usize = 30;
const DEFAULT_Z_WINDOW: usize = 50;
const DEFAULT_PCT_VOL_LOOKBACK: usize = 100;
const DEFAULT_PCT_MIN_SIGMA: f64 = 0.1;
const DEFAULT_ABS_VOL_LOOKBACK: usize = 100;
const DAY_MS: i64 = 86_400_000;

#[derive(Debug, Error)]
pub enum CudaVwapDeviationOscillatorError {
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
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
}

pub struct VwapDeviationOscillatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl VwapDeviationOscillatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct VwapDeviationOscillatorDeviceOutputs {
    pub osc: VwapDeviationOscillatorDeviceArrayF64,
    pub std1: VwapDeviationOscillatorDeviceArrayF64,
    pub std2: VwapDeviationOscillatorDeviceArrayF64,
    pub std3: VwapDeviationOscillatorDeviceArrayF64,
}

impl VwapDeviationOscillatorDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.osc.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.osc.cols
    }
}

pub struct CudaVwapDeviationOscillatorBatchResult {
    pub outputs: VwapDeviationOscillatorDeviceOutputs,
    pub combos: Vec<VwapDeviationOscillatorParams>,
}

pub struct CudaVwapDeviationOscillator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct BaseKey {
    session_mode: VwapDeviationSessionMode,
    rolling_period: usize,
    rolling_days: usize,
    use_close: bool,
}

struct BaseSeries {
    resid_abs: Vec<f64>,
    resid_pct: Vec<f64>,
}

#[derive(Clone, Copy)]
enum DeviationModeKind {
    Absolute = 0,
    Percent = 1,
    ZScore = 2,
}

struct RowConfig {
    key: BaseKey,
    mode: DeviationModeKind,
    window: usize,
    guard: f64,
}

#[inline]
fn price_ref(high: f64, low: f64, close: f64, use_close: bool) -> f64 {
    if use_close {
        close
    } else if high.is_finite() && low.is_finite() && close.is_finite() {
        (high + low + close) / 3.0
    } else {
        f64::NAN
    }
}

#[inline]
fn period_id(timestamp_ms: i64, session_mode: VwapDeviationSessionMode) -> i64 {
    let sec = timestamp_ms / 1_000;
    match session_mode {
        VwapDeviationSessionMode::FourHours => sec / 3_600 / 4,
        VwapDeviationSessionMode::Daily => sec / 86_400,
        VwapDeviationSessionMode::Weekly => (sec / 86_400 + 3) / 7,
        VwapDeviationSessionMode::RollingBars | VwapDeviationSessionMode::RollingDays => 0,
    }
}

fn compute_base_series(
    timestamps: &[i64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    key: BaseKey,
) -> BaseSeries {
    let len = close.len();
    let mut resid_abs = vec![f64::NAN; len];
    let mut resid_pct = vec![f64::NAN; len];

    match key.session_mode {
        VwapDeviationSessionMode::RollingBars => {
            let period = key.rolling_period;
            let mut entries = vec![(0.0, 0.0); period];
            let mut head = 0usize;
            let mut count = 0usize;
            let mut sum_pv = 0.0;
            let mut sum_vol = 0.0;
            for i in 0..len {
                let pr = price_ref(high[i], low[i], close[i], key.use_close);
                let contrib = if pr.is_finite() && volume[i].is_finite() {
                    (pr * volume[i], volume[i])
                } else {
                    (0.0, 0.0)
                };
                if count == period {
                    let (old_pv, old_vol) = entries[head];
                    sum_pv -= old_pv;
                    sum_vol -= old_vol;
                } else {
                    count += 1;
                }
                entries[head] = contrib;
                sum_pv += contrib.0;
                sum_vol += contrib.1;
                head += 1;
                if head == period {
                    head = 0;
                }
                let vwap = if sum_vol != 0.0 {
                    sum_pv / sum_vol
                } else {
                    f64::NAN
                };
                if pr.is_finite() && vwap.is_finite() {
                    resid_abs[i] = pr - vwap;
                    if vwap != 0.0 {
                        resid_pct[i] = 100.0 * (pr / vwap - 1.0);
                    }
                }
            }
        }
        VwapDeviationSessionMode::RollingDays => {
            let mut entries: VecDeque<(i64, f64, f64)> = VecDeque::new();
            let mut sum_pv = 0.0;
            let mut sum_vol = 0.0;
            let days_ms = (key.rolling_days as i64).saturating_mul(DAY_MS);
            for i in 0..len {
                let pr = price_ref(high[i], low[i], close[i], key.use_close);
                let contrib = if pr.is_finite() && volume[i].is_finite() {
                    (pr * volume[i], volume[i])
                } else {
                    (0.0, 0.0)
                };
                entries.push_back((timestamps[i], contrib.0, contrib.1));
                sum_pv += contrib.0;
                sum_vol += contrib.1;
                let cutoff = timestamps[i].saturating_sub(days_ms);
                while entries
                    .front()
                    .map(|(ts, _, _)| *ts < cutoff)
                    .unwrap_or(false)
                {
                    if let Some((_, old_pv, old_vol)) = entries.pop_front() {
                        sum_pv -= old_pv;
                        sum_vol -= old_vol;
                    }
                }
                let vwap = if sum_vol != 0.0 {
                    sum_pv / sum_vol
                } else {
                    f64::NAN
                };
                if pr.is_finite() && vwap.is_finite() {
                    resid_abs[i] = pr - vwap;
                    if vwap != 0.0 {
                        resid_pct[i] = 100.0 * (pr / vwap - 1.0);
                    }
                }
            }
        }
        VwapDeviationSessionMode::FourHours
        | VwapDeviationSessionMode::Daily
        | VwapDeviationSessionMode::Weekly => {
            let mut last_id: Option<i64> = None;
            let mut sum_pv = 0.0;
            let mut sum_vol = 0.0;
            for i in 0..len {
                let id = period_id(timestamps[i], key.session_mode);
                if last_id.map(|prev| prev != id).unwrap_or(true) {
                    last_id = Some(id);
                    sum_pv = 0.0;
                    sum_vol = 0.0;
                }
                let pr = price_ref(high[i], low[i], close[i], key.use_close);
                if pr.is_finite() && volume[i].is_finite() {
                    sum_pv += pr * volume[i];
                    sum_vol += volume[i];
                }
                let vwap = if sum_vol != 0.0 {
                    sum_pv / sum_vol
                } else {
                    f64::NAN
                };
                if pr.is_finite() && vwap.is_finite() {
                    resid_abs[i] = pr - vwap;
                    if vwap != 0.0 {
                        resid_pct[i] = 100.0 * (pr / vwap - 1.0);
                    }
                }
            }
        }
    }

    BaseSeries {
        resid_abs,
        resid_pct,
    }
}

impl CudaVwapDeviationOscillator {
    pub fn new(device_id: usize) -> Result<Self, CudaVwapDeviationOscillatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("vwap_deviation_oscillator_kernel")?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
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

    pub fn synchronize(&self) -> Result<(), CudaVwapDeviationOscillatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn validate_inputs(
        timestamps: &[i64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<(), CudaVwapDeviationOscillatorError> {
        if timestamps.is_empty()
            || high.is_empty()
            || low.is_empty()
            || close.is_empty()
            || volume.is_empty()
        {
            return Err(CudaVwapDeviationOscillatorError::InvalidInput(
                "empty input".into(),
            ));
        }
        if timestamps.len() != high.len()
            || timestamps.len() != low.len()
            || timestamps.len() != close.len()
            || timestamps.len() != volume.len()
        {
            return Err(CudaVwapDeviationOscillatorError::InvalidInput(format!(
                "input length mismatch: timestamps={}, high={}, low={}, close={}, volume={}",
                timestamps.len(),
                high.len(),
                low.len(),
                close.len(),
                volume.len()
            )));
        }
        if !high
            .iter()
            .zip(low.iter())
            .zip(close.iter())
            .zip(volume.iter())
            .any(|(((h, l), c), v)| {
                h.is_finite() || l.is_finite() || c.is_finite() || v.is_finite()
            })
        {
            return Err(CudaVwapDeviationOscillatorError::InvalidInput(
                "all values are NaN".into(),
            ));
        }
        Ok(())
    }

    fn validate_launch(
        &self,
        grid: GridSize,
        block: BlockSize,
    ) -> Result<(), CudaVwapDeviationOscillatorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaVwapDeviationOscillatorError::LaunchConfigTooLarge {
                gx: grid.x,
                gy: grid.y,
                gz: grid.z,
                bx: block.x,
                by: block.y,
                bz: block.z,
            });
        }
        Ok(())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaVwapDeviationOscillatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaVwapDeviationOscillatorError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            }
        }
        Ok(())
    }

    fn row_config(
        combo: &VwapDeviationOscillatorParams,
    ) -> Result<RowConfig, CudaVwapDeviationOscillatorError> {
        let session_mode = combo
            .session_mode
            .unwrap_or(VwapDeviationSessionMode::RollingBars);
        let rolling_period = combo.rolling_period.unwrap_or(DEFAULT_ROLLING_PERIOD);
        let rolling_days = combo.rolling_days.unwrap_or(DEFAULT_ROLLING_DAYS);
        let use_close = combo.use_close.unwrap_or(false);
        let deviation_mode = combo.deviation_mode.unwrap_or(VwapDeviationMode::Absolute);
        let z_window = combo.z_window.unwrap_or(DEFAULT_Z_WINDOW);
        let pct_vol_lookback = combo.pct_vol_lookback.unwrap_or(DEFAULT_PCT_VOL_LOOKBACK);
        let pct_min_sigma = combo.pct_min_sigma.unwrap_or(DEFAULT_PCT_MIN_SIGMA);
        let abs_vol_lookback = combo.abs_vol_lookback.unwrap_or(DEFAULT_ABS_VOL_LOOKBACK);

        if rolling_period == 0 {
            return Err(CudaVwapDeviationOscillatorError::InvalidInput(format!(
                "invalid rolling_period: {rolling_period}"
            )));
        }
        if rolling_days == 0 {
            return Err(CudaVwapDeviationOscillatorError::InvalidInput(format!(
                "invalid rolling_days: {rolling_days}"
            )));
        }
        if z_window < 5 {
            return Err(CudaVwapDeviationOscillatorError::InvalidInput(format!(
                "invalid z_window: {z_window}"
            )));
        }
        if pct_vol_lookback < 10 {
            return Err(CudaVwapDeviationOscillatorError::InvalidInput(format!(
                "invalid pct_vol_lookback: {pct_vol_lookback}"
            )));
        }
        if !pct_min_sigma.is_finite() || pct_min_sigma < 0.01 {
            return Err(CudaVwapDeviationOscillatorError::InvalidInput(format!(
                "invalid pct_min_sigma: {pct_min_sigma}"
            )));
        }
        if abs_vol_lookback < 10 {
            return Err(CudaVwapDeviationOscillatorError::InvalidInput(format!(
                "invalid abs_vol_lookback: {abs_vol_lookback}"
            )));
        }

        let (mode, window, guard) = match deviation_mode {
            VwapDeviationMode::Absolute => (DeviationModeKind::Absolute, abs_vol_lookback, 1.0),
            VwapDeviationMode::Percent => {
                (DeviationModeKind::Percent, pct_vol_lookback, pct_min_sigma)
            }
            VwapDeviationMode::ZScore => (DeviationModeKind::ZScore, z_window, 0.0),
        };

        Ok(RowConfig {
            key: BaseKey {
                session_mode,
                rolling_period,
                rolling_days,
                use_close,
            },
            mode,
            window,
            guard,
        })
    }

    pub fn batch_dev(
        &self,
        timestamps: &[i64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
        sweep: &VwapDeviationOscillatorBatchRange,
    ) -> Result<CudaVwapDeviationOscillatorBatchResult, CudaVwapDeviationOscillatorError> {
        Self::validate_inputs(timestamps, high, low, close, volume)?;

        let combos = expand_grid(sweep)
            .map_err(|e| CudaVwapDeviationOscillatorError::InvalidInput(e.to_string()))?;
        if combos.is_empty() {
            return Err(CudaVwapDeviationOscillatorError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = close.len();
        let mut row_configs = Vec::with_capacity(rows);
        let mut max_window = 0usize;
        for combo in &combos {
            let row = Self::row_config(combo)?;
            max_window = max_window.max(row.window);
            row_configs.push(row);
        }

        let total = rows.checked_mul(cols).ok_or_else(|| {
            CudaVwapDeviationOscillatorError::InvalidInput("rows*cols overflow".into())
        })?;
        let mut source_values = vec![f64::NAN; total];
        let mut windows = Vec::with_capacity(rows);
        let mut guards = Vec::with_capacity(rows);
        let mut modes = Vec::with_capacity(rows);
        let mut base_map: HashMap<BaseKey, BaseSeries> = HashMap::new();

        for (row, cfg) in row_configs.iter().enumerate() {
            base_map.entry(cfg.key).or_insert_with(|| {
                compute_base_series(timestamps, high, low, close, volume, cfg.key)
            });
            let base = base_map.get(&cfg.key).expect("base series");
            let dst = &mut source_values[row * cols..(row + 1) * cols];
            match cfg.mode {
                DeviationModeKind::Absolute | DeviationModeKind::ZScore => {
                    dst.copy_from_slice(&base.resid_abs);
                }
                DeviationModeKind::Percent => {
                    dst.copy_from_slice(&base.resid_pct);
                }
            }
            windows.push(i32::try_from(cfg.window).map_err(|_| {
                CudaVwapDeviationOscillatorError::InvalidInput(format!(
                    "window out of range: {}",
                    cfg.window
                ))
            })?);
            guards.push(cfg.guard);
            modes.push(cfg.mode as i32);
        }

        let input_bytes = total
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaVwapDeviationOscillatorError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                rows.checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                rows.checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaVwapDeviationOscillatorError::InvalidInput("params bytes overflow".into())
            })?;
        let scratch_elems = rows.checked_mul(max_window).ok_or_else(|| {
            CudaVwapDeviationOscillatorError::InvalidInput("scratch overflow".into())
        })?;
        let scratch_bytes = scratch_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaVwapDeviationOscillatorError::InvalidInput("scratch bytes overflow".into())
            })?;
        let output_bytes = total
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| {
                CudaVwapDeviationOscillatorError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(scratch_bytes))
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaVwapDeviationOscillatorError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_source_values = DeviceBuffer::from_slice(&source_values)?;
        let d_modes = DeviceBuffer::from_slice(&modes)?;
        let d_windows = DeviceBuffer::from_slice(&windows)?;
        let d_guards = DeviceBuffer::from_slice(&guards)?;
        let d_scratch_values = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_elems)? };
        let d_out_osc = unsafe { DeviceBuffer::<f64>::uninitialized(total)? };
        let d_out_std1 = unsafe { DeviceBuffer::<f64>::uninitialized(total)? };
        let d_out_std2 = unsafe { DeviceBuffer::<f64>::uninitialized(total)? };
        let d_out_std3 = unsafe { DeviceBuffer::<f64>::uninitialized(total)? };

        let func = self
            .module
            .get_function("vwap_deviation_oscillator_batch_f64")
            .map_err(|_| CudaVwapDeviationOscillatorError::MissingKernelSymbol {
                name: "vwap_deviation_oscillator_batch_f64",
            })?;
        let grid_x = ((rows as u32) + VWAP_DEVIATION_OSCILLATOR_BLOCK_X - 1)
            / VWAP_DEVIATION_OSCILLATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(VWAP_DEVIATION_OSCILLATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_source_values.as_device_ptr(),
                cols as i32,
                d_modes.as_device_ptr(),
                d_windows.as_device_ptr(),
                d_guards.as_device_ptr(),
                rows as i32,
                max_window as i32,
                d_scratch_values.as_device_ptr(),
                d_out_osc.as_device_ptr(),
                d_out_std1.as_device_ptr(),
                d_out_std2.as_device_ptr(),
                d_out_std3.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaVwapDeviationOscillatorBatchResult {
            outputs: VwapDeviationOscillatorDeviceOutputs {
                osc: VwapDeviationOscillatorDeviceArrayF64 {
                    buf: d_out_osc,
                    rows,
                    cols,
                },
                std1: VwapDeviationOscillatorDeviceArrayF64 {
                    buf: d_out_std1,
                    rows,
                    cols,
                },
                std2: VwapDeviationOscillatorDeviceArrayF64 {
                    buf: d_out_std2,
                    rows,
                    cols,
                },
                std3: VwapDeviationOscillatorDeviceArrayF64 {
                    buf: d_out_std3,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
