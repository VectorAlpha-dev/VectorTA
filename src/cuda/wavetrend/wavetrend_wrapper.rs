#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::wavetrend::{WavetrendBatchRange, WavetrendParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{CopyDestination, DeviceBuffer};
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaWavetrendError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaWavetrendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaWavetrendError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaWavetrendError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaWavetrendError {}

pub struct CudaWavetrend {
    module: Module,
    stream: Stream,
    _context: Context,
}

pub struct CudaWavetrendBatch {
    pub wt1: DeviceArrayF32,
    pub wt2: DeviceArrayF32,
    pub wt_diff: DeviceArrayF32,
    pub combos: Vec<WavetrendParams>,
}

struct PreparedBatch {
    combos: Vec<WavetrendParams>,
    first_valid: usize,
    series_len: usize,
}

impl CudaWavetrend {
    pub fn new(device_id: usize) -> Result<Self, CudaWavetrendError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;

        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/wavetrend_kernel.ptx"));
        let module =
            Module::from_ptx(ptx, &[]).map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    pub fn wavetrend_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &WavetrendBatchRange,
    ) -> Result<CudaWavetrendBatch, CudaWavetrendError> {
        let PreparedBatch {
            combos,
            first_valid,
            series_len,
        } = Self::prepare_batch_inputs(data_f32, sweep)?;
        let rows = combos.len();

        let d_prices = DeviceBuffer::from_slice(data_f32)
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;

        let (channels, averages, mas, factors) = Self::build_param_arrays(&combos)?;
        let d_channels = DeviceBuffer::from_slice(&channels)
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        let d_averages = DeviceBuffer::from_slice(&averages)
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        let d_mas =
            DeviceBuffer::from_slice(&mas).map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        let d_factors = DeviceBuffer::from_slice(&factors)
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;

        let mut d_wt1: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(rows * series_len) }
                .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        let mut d_wt2: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(rows * series_len) }
                .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        let mut d_wt_diff: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(rows * series_len) }
                .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;

        self.launch_kernel(
            &d_prices,
            &d_channels,
            &d_averages,
            &d_mas,
            &d_factors,
            first_valid,
            series_len,
            rows,
            &mut d_wt1,
            &mut d_wt2,
            &mut d_wt_diff,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;

        Ok(CudaWavetrendBatch {
            wt1: DeviceArrayF32 {
                buf: d_wt1,
                rows,
                cols: series_len,
            },
            wt2: DeviceArrayF32 {
                buf: d_wt2,
                rows,
                cols: series_len,
            },
            wt_diff: DeviceArrayF32 {
                buf: d_wt_diff,
                rows,
                cols: series_len,
            },
            combos,
        })
    }

    pub fn wavetrend_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &WavetrendBatchRange,
        out_wt1: &mut [f32],
        out_wt2: &mut [f32],
        out_wt_diff: &mut [f32],
    ) -> Result<(usize, usize, Vec<WavetrendParams>), CudaWavetrendError> {
        let batch = self.wavetrend_batch_dev(data_f32, sweep)?;
        let rows = batch.wt1.rows;
        let cols = batch.wt1.cols;
        let expected = rows * cols;
        if out_wt1.len() != expected || out_wt2.len() != expected || out_wt_diff.len() != expected {
            return Err(CudaWavetrendError::InvalidInput(format!(
                "output slices have wrong length (expected {})",
                expected
            )));
        }

        batch
            .wt1
            .buf
            .copy_to(out_wt1)
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        batch
            .wt2
            .buf
            .copy_to(out_wt2)
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        batch
            .wt_diff
            .buf
            .copy_to(out_wt_diff)
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;

        Ok((rows, cols, batch.combos))
    }

    pub fn wavetrend_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        combos: &[WavetrendParams],
        first_valid: usize,
        series_len: usize,
        d_wt1: &mut DeviceBuffer<f32>,
        d_wt2: &mut DeviceBuffer<f32>,
        d_wt_diff: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWavetrendError> {
        if combos.is_empty() {
            return Err(CudaWavetrendError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        if series_len == 0 {
            return Err(CudaWavetrendError::InvalidInput(
                "series_len is zero".into(),
            ));
        }
        if d_prices.len() != series_len {
            return Err(CudaWavetrendError::InvalidInput(format!(
                "price buffer len {} != series_len {}",
                d_prices.len(),
                series_len
            )));
        }

        let (channels, averages, mas, factors) = Self::build_param_arrays(combos)?;
        let d_channels = DeviceBuffer::from_slice(&channels)
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        let d_averages = DeviceBuffer::from_slice(&averages)
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        let d_mas =
            DeviceBuffer::from_slice(&mas).map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        let d_factors = DeviceBuffer::from_slice(&factors)
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;

        self.launch_kernel(
            d_prices,
            &d_channels,
            &d_averages,
            &d_mas,
            &d_factors,
            first_valid,
            series_len,
            combos.len(),
            d_wt1,
            d_wt2,
            d_wt_diff,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_channels: &DeviceBuffer<i32>,
        d_averages: &DeviceBuffer<i32>,
        d_mas: &DeviceBuffer<i32>,
        d_factors: &DeviceBuffer<f32>,
        first_valid: usize,
        series_len: usize,
        rows: usize,
        d_wt1: &mut DeviceBuffer<f32>,
        d_wt2: &mut DeviceBuffer<f32>,
        d_wt_diff: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWavetrendError> {
        if series_len == 0 {
            return Err(CudaWavetrendError::InvalidInput(
                "series_len is zero".into(),
            ));
        }
        if d_prices.len() != series_len {
            return Err(CudaWavetrendError::InvalidInput(format!(
                "price buffer len {} != series_len {}",
                d_prices.len(),
                series_len
            )));
        }
        if d_channels.len() != rows
            || d_averages.len() != rows
            || d_mas.len() != rows
            || d_factors.len() != rows
        {
            return Err(CudaWavetrendError::InvalidInput(
                "parameter buffers must match number of combinations".into(),
            ));
        }
        let expected = rows * series_len;
        if d_wt1.len() != expected || d_wt2.len() != expected || d_wt_diff.len() != expected {
            return Err(CudaWavetrendError::InvalidInput(format!(
                "output buffer mismatch: expected {} entries per output",
                expected
            )));
        }
        if series_len > i32::MAX as usize {
            return Err(CudaWavetrendError::InvalidInput(
                "series length exceeds i32::MAX".into(),
            ));
        }
        if rows > i32::MAX as usize {
            return Err(CudaWavetrendError::InvalidInput(
                "row count exceeds i32::MAX".into(),
            ));
        }

        let func = self
            .module
            .get_function("wavetrend_batch_f32")
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;

        let block_x: u32 = 128;
        let mut grid_x = ((rows as u32) + block_x - 1) / block_x;
        if grid_x == 0 {
            grid_x = 1;
        }
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut len_i = series_len as i32;
            let mut first_i = first_valid as i32;
            let mut rows_i = rows as i32;
            let mut ch_ptr = d_channels.as_device_ptr().as_raw();
            let mut avg_ptr = d_averages.as_device_ptr().as_raw();
            let mut ma_ptr = d_mas.as_device_ptr().as_raw();
            let mut factor_ptr = d_factors.as_device_ptr().as_raw();
            let mut wt1_ptr = d_wt1.as_device_ptr().as_raw();
            let mut wt2_ptr = d_wt2.as_device_ptr().as_raw();
            let mut diff_ptr = d_wt_diff.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut ch_ptr as *mut _ as *mut c_void,
                &mut avg_ptr as *mut _ as *mut c_void,
                &mut ma_ptr as *mut _ as *mut c_void,
                &mut factor_ptr as *mut _ as *mut c_void,
                &mut wt1_ptr as *mut _ as *mut c_void,
                &mut wt2_ptr as *mut _ as *mut c_void,
                &mut diff_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    fn prepare_batch_inputs(
        data: &[f32],
        sweep: &WavetrendBatchRange,
    ) -> Result<PreparedBatch, CudaWavetrendError> {
        if data.is_empty() {
            return Err(CudaWavetrendError::InvalidInput("empty data".into()));
        }
        let first_valid = data
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaWavetrendError::InvalidInput("all values are NaN".into()))?;
        let series_len = data.len();
        let combos = Self::expand_range(sweep);
        if combos.is_empty() {
            return Err(CudaWavetrendError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        if series_len > i32::MAX as usize {
            return Err(CudaWavetrendError::InvalidInput(
                "series length exceeds i32::MAX (unsupported)".into(),
            ));
        }
        if combos.len() > i32::MAX as usize {
            return Err(CudaWavetrendError::InvalidInput(
                "combination count exceeds i32::MAX (unsupported)".into(),
            ));
        }

        for (idx, combo) in combos.iter().enumerate() {
            let ch = combo.channel_length.unwrap_or(0);
            let avg = combo.average_length.unwrap_or(0);
            let ma = combo.ma_length.unwrap_or(0);
            if ch == 0 || avg == 0 || ma == 0 {
                return Err(CudaWavetrendError::InvalidInput(format!(
                    "invalid periods at combo {} (ch={}, avg={}, ma={})",
                    idx, ch, avg, ma
                )));
            }
            if ch > series_len || avg > series_len || ma > series_len {
                return Err(CudaWavetrendError::InvalidInput(format!(
                    "period exceeds series length at combo {}",
                    idx
                )));
            }
            let needed = ch.max(avg).max(ma);
            let valid = series_len - first_valid;
            if valid < needed {
                return Err(CudaWavetrendError::InvalidInput(format!(
                    "not enough valid data for combo {} (needed {}, valid {})",
                    idx, needed, valid
                )));
            }
        }

        Ok(PreparedBatch {
            combos,
            first_valid,
            series_len,
        })
    }

    fn build_param_arrays(
        combos: &[WavetrendParams],
    ) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>, Vec<f32>), CudaWavetrendError> {
        let mut channels = Vec::with_capacity(combos.len());
        let mut averages = Vec::with_capacity(combos.len());
        let mut mas = Vec::with_capacity(combos.len());
        let mut factors = Vec::with_capacity(combos.len());
        for combo in combos {
            channels.push(combo.channel_length.unwrap_or(0) as i32);
            averages.push(combo.average_length.unwrap_or(0) as i32);
            mas.push(combo.ma_length.unwrap_or(0) as i32);
            factors.push(combo.factor.unwrap_or(0.015) as f32);
        }
        Ok((channels, averages, mas, factors))
    }

    fn expand_range(sweep: &WavetrendBatchRange) -> Vec<WavetrendParams> {
        fn axis_usize(axis: (usize, usize, usize)) -> Vec<usize> {
            let (start, end, step) = axis;
            if step == 0 || start == end {
                return vec![start];
            }
            (start..=end).step_by(step).collect()
        }
        fn axis_f64(axis: (f64, f64, f64)) -> Vec<f64> {
            let (start, end, step) = axis;
            if step.abs() < f64::EPSILON || (start - end).abs() < f64::EPSILON {
                return vec![start];
            }
            let mut out = Vec::new();
            let mut v = start;
            while v <= end + f64::EPSILON {
                out.push(v);
                v += step;
            }
            out
        }

        let channels = axis_usize(sweep.channel_length);
        let averages = axis_usize(sweep.average_length);
        let mas = axis_usize(sweep.ma_length);
        let factors = axis_f64(sweep.factor);

        let mut combos =
            Vec::with_capacity(channels.len() * averages.len() * mas.len() * factors.len());
        for &ch in &channels {
            for &avg in &averages {
                for &ma in &mas {
                    for &f in &factors {
                        combos.push(WavetrendParams {
                            channel_length: Some(ch),
                            average_length: Some(avg),
                            ma_length: Some(ma),
                            factor: Some(f),
                        });
                    }
                }
            }
        }
        combos
    }
}

// ---------- Bench profiles (batch only) ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        // 3 outputs
        let out_bytes = 3 * ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct WtBatchState {
        cuda: CudaWavetrend,
        price: Vec<f32>,
        sweep: WavetrendBatchRange,
    }
    impl CudaBenchState for WtBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .wavetrend_batch_dev(&self.price, &self.sweep)
                .expect("wavetrend batch");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaWavetrend::new(0).expect("cuda wavetrend");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = WavetrendBatchRange {
            channel_length: (10, 10 + PARAM_SWEEP - 1, 1),
            average_length: (21, 21, 0),
            ma_length: (4, 4, 0),
            factor: (0.015, 0.015, 0.0),
        };
        Box::new(WtBatchState { cuda, price, sweep })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "wavetrend",
            "one_series_many_params",
            "wavetrend_cuda_batch_dev",
            "1m_x_250",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
