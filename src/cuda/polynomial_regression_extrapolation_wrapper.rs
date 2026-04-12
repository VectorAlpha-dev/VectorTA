#![cfg(feature = "cuda")]

use crate::indicators::polynomial_regression_extrapolation::{
    expand_grid, PolynomialRegressionExtrapolationBatchRange,
    PolynomialRegressionExtrapolationParams,
};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::sync::Arc;
use thiserror::Error;

const POLYNOMIAL_REGRESSION_EXTRAPOLATION_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const MAX_DEGREE: usize = 8;
const SINGULAR_EPSILON: f64 = 1e-12;

#[derive(Debug, Error)]
pub enum CudaPolynomialRegressionExtrapolationError {
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

pub struct PolynomialRegressionExtrapolationDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl PolynomialRegressionExtrapolationDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaPolynomialRegressionExtrapolationBatchResult {
    pub outputs: PolynomialRegressionExtrapolationDeviceArrayF64,
    pub combos: Vec<PolynomialRegressionExtrapolationParams>,
}

pub struct CudaPolynomialRegressionExtrapolation {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaPolynomialRegressionExtrapolation {
    pub fn new(device_id: usize) -> Result<Self, CudaPolynomialRegressionExtrapolationError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module =
            crate::load_cuda_embedded_module!("polynomial_regression_extrapolation_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaPolynomialRegressionExtrapolationError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaPolynomialRegressionExtrapolationError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaPolynomialRegressionExtrapolationError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            }
        }
        Ok(())
    }

    fn validate_launch(
        &self,
        grid: GridSize,
        block: BlockSize,
    ) -> Result<(), CudaPolynomialRegressionExtrapolationError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(
                CudaPolynomialRegressionExtrapolationError::LaunchConfigTooLarge {
                    gx: grid.x,
                    gy: grid.y,
                    gz: grid.z,
                    bx: block.x,
                    by: block.y,
                    bz: block.z,
                },
            );
        }
        Ok(())
    }

    pub fn batch_dev(
        &self,
        data: &[f64],
        sweep: &PolynomialRegressionExtrapolationBatchRange,
    ) -> Result<
        CudaPolynomialRegressionExtrapolationBatchResult,
        CudaPolynomialRegressionExtrapolationError,
    > {
        if data.is_empty() {
            return Err(CudaPolynomialRegressionExtrapolationError::InvalidInput(
                "empty input".into(),
            ));
        }

        let first = data
            .iter()
            .position(|value| !value.is_nan())
            .ok_or_else(|| {
                CudaPolynomialRegressionExtrapolationError::InvalidInput(
                    "all values are NaN".into(),
                )
            })?;
        let valid = data.len() - first;

        let combos = expand_grid(sweep).map_err(|err| {
            CudaPolynomialRegressionExtrapolationError::InvalidInput(err.to_string())
        })?;
        if combos.is_empty() {
            return Err(CudaPolynomialRegressionExtrapolationError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = data.len();
        let mut max_length = 0usize;
        let mut lengths = Vec::with_capacity(rows);
        let mut row_weights = Vec::with_capacity(rows);

        for combo in &combos {
            let length = combo.length.unwrap_or(100);
            let extrapolate = combo.extrapolate.unwrap_or(10);
            let degree = combo.degree.unwrap_or(3);
            if length == 0 || length > data.len() {
                return Err(CudaPolynomialRegressionExtrapolationError::InvalidInput(
                    format!("invalid length: length={length}, data_len={}", data.len()),
                ));
            }
            if valid < length {
                return Err(CudaPolynomialRegressionExtrapolationError::InvalidInput(
                    format!("not enough valid data: needed={length}, valid={valid}"),
                ));
            }
            let weights = build_forecast_weights(length, extrapolate, degree)
                .map_err(CudaPolynomialRegressionExtrapolationError::InvalidInput)?;
            max_length = max_length.max(length);
            lengths.push(length as i32);
            row_weights.push(weights);
        }

        let flat_weights_len = rows.checked_mul(max_length).ok_or_else(|| {
            CudaPolynomialRegressionExtrapolationError::InvalidInput(
                "rows*max_length overflow".into(),
            )
        })?;
        let mut flat_weights = vec![0.0f64; flat_weights_len];
        for (row, weights) in row_weights.iter().enumerate() {
            let start = row * max_length;
            flat_weights[start..start + weights.len()].copy_from_slice(weights);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaPolynomialRegressionExtrapolationError::InvalidInput(
                    "input bytes overflow".into(),
                )
            })?;
        let params_bytes = lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaPolynomialRegressionExtrapolationError::InvalidInput(
                    "params bytes overflow".into(),
                )
            })?;
        let weights_bytes = flat_weights
            .len()
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaPolynomialRegressionExtrapolationError::InvalidInput(
                    "weights bytes overflow".into(),
                )
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaPolynomialRegressionExtrapolationError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaPolynomialRegressionExtrapolationError::InvalidInput(
                    "output bytes overflow".into(),
                )
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(weights_bytes))
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaPolynomialRegressionExtrapolationError::InvalidInput(
                    "required bytes overflow".into(),
                )
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_weights = DeviceBuffer::from_slice(&flat_weights)?;
        let d_out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("polynomial_regression_extrapolation_batch_f64")
            .map_err(
                |_| CudaPolynomialRegressionExtrapolationError::MissingKernelSymbol {
                    name: "polynomial_regression_extrapolation_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + POLYNOMIAL_REGRESSION_EXTRAPOLATION_BLOCK_X - 1)
            / POLYNOMIAL_REGRESSION_EXTRAPOLATION_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(POLYNOMIAL_REGRESSION_EXTRAPOLATION_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                rows as i32,
                max_length as i32,
                d_weights.as_device_ptr(),
                d_out.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaPolynomialRegressionExtrapolationBatchResult {
            outputs: PolynomialRegressionExtrapolationDeviceArrayF64 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}

fn solve_dense_system_in_place(matrix: &mut [f64], rhs: &mut [f64], n: usize) -> Result<(), ()> {
    for pivot_col in 0..n {
        let mut pivot_row = pivot_col;
        let mut pivot_abs = matrix[pivot_col * n + pivot_col].abs();
        for row in (pivot_col + 1)..n {
            let candidate = matrix[row * n + pivot_col].abs();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }
        if pivot_abs <= SINGULAR_EPSILON {
            return Err(());
        }
        if pivot_row != pivot_col {
            for col in pivot_col..n {
                matrix.swap(pivot_col * n + col, pivot_row * n + col);
            }
            rhs.swap(pivot_col, pivot_row);
        }
        let pivot = matrix[pivot_col * n + pivot_col];
        for row in (pivot_col + 1)..n {
            let factor = matrix[row * n + pivot_col] / pivot;
            if factor == 0.0 {
                continue;
            }
            matrix[row * n + pivot_col] = 0.0;
            for col in (pivot_col + 1)..n {
                matrix[row * n + col] -= factor * matrix[pivot_col * n + col];
            }
            rhs[row] -= factor * rhs[pivot_col];
        }
    }

    for row in (0..n).rev() {
        let mut acc = rhs[row];
        for col in (row + 1)..n {
            acc -= matrix[row * n + col] * rhs[col];
        }
        let pivot = matrix[row * n + row];
        if pivot.abs() <= SINGULAR_EPSILON {
            return Err(());
        }
        rhs[row] = acc / pivot;
    }
    Ok(())
}

fn build_forecast_weights(
    length: usize,
    extrapolate: usize,
    degree: usize,
) -> Result<Vec<f64>, String> {
    if degree > MAX_DEGREE {
        return Err(format!("invalid degree: degree={degree}, max={MAX_DEGREE}"));
    }
    if degree + 1 > length {
        return Err(format!(
            "degree exceeds length: degree={degree}, length={length}"
        ));
    }

    let order_count = degree + 1;
    let mut normal = vec![0.0; order_count * order_count];
    for row in 0..order_count {
        for col in 0..order_count {
            let power = row + col;
            let mut sum = 0.0;
            for x in 0..length {
                sum += (x as f64).powi(power as i32);
            }
            normal[row * order_count + col] = sum;
        }
    }

    let x_eval = -(extrapolate as f64);
    let mut rhs = vec![0.0; order_count];
    for (power, value) in rhs.iter_mut().enumerate() {
        *value = x_eval.powi(power as i32);
    }
    solve_dense_system_in_place(&mut normal, &mut rhs, order_count)
        .map_err(|_| format!("singular polynomial fit for length={length}, degree={degree}"))?;

    let mut weights = vec![0.0; length];
    for (x, weight) in weights.iter_mut().enumerate() {
        let xf = x as f64;
        let mut acc = 0.0f64;
        for power in (0..order_count).rev() {
            acc = acc.mul_add(xf, rhs[power]);
        }
        *weight = acc;
    }
    Ok(weights)
}
