#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "python")]
use pyo3::wrap_pyfunction;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use std::convert::AsRef;
use std::mem::ManuallyDrop;
use thiserror::Error;

#[inline(always)]
fn advance_decline_line_source<'a>(candles: &'a Candles, source: &str) -> &'a [f64] {
    match source {
        "open" => &candles.open,
        "high" => &candles.high,
        "low" => &candles.low,
        "close" => &candles.close,
        "volume" => &candles.volume,
        _ => source_type(candles, source),
    }
}

impl<'a> AsRef<[f64]> for AdvanceDeclineLineInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            AdvanceDeclineLineData::Slice(slice) => slice,
            AdvanceDeclineLineData::Candles { candles, source } => {
                advance_decline_line_source(candles, source)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum AdvanceDeclineLineData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct AdvanceDeclineLineOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct AdvanceDeclineLineParams;

#[derive(Debug, Clone)]
pub struct AdvanceDeclineLineInput<'a> {
    pub data: AdvanceDeclineLineData<'a>,
    pub params: AdvanceDeclineLineParams,
}

impl<'a> AdvanceDeclineLineInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: AdvanceDeclineLineParams,
    ) -> Self {
        Self {
            data: AdvanceDeclineLineData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: AdvanceDeclineLineParams) -> Self {
        Self {
            data: AdvanceDeclineLineData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", AdvanceDeclineLineParams)
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct AdvanceDeclineLineBuilder {
    kernel: Kernel,
}

impl AdvanceDeclineLineBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<AdvanceDeclineLineOutput, AdvanceDeclineLineError> {
        let input =
            AdvanceDeclineLineInput::from_candles(candles, source, AdvanceDeclineLineParams);
        advance_decline_line_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<AdvanceDeclineLineOutput, AdvanceDeclineLineError> {
        let input = AdvanceDeclineLineInput::from_slice(data, AdvanceDeclineLineParams);
        advance_decline_line_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<AdvanceDeclineLineStream, AdvanceDeclineLineError> {
        let _ = self.kernel;
        AdvanceDeclineLineStream::try_new()
    }
}

#[derive(Debug, Error)]
pub enum AdvanceDeclineLineError {
    #[error("advance_decline_line: Input data slice is empty.")]
    EmptyInputData,
    #[error("advance_decline_line: All values are NaN.")]
    AllValuesNaN,
    #[error("advance_decline_line: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("advance_decline_line: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("advance_decline_line: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone, Default)]
pub struct AdvanceDeclineLineStream {
    started: bool,
    sum: f64,
}

impl AdvanceDeclineLineStream {
    #[inline]
    pub fn try_new() -> Result<Self, AdvanceDeclineLineError> {
        Ok(Self::default())
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.started = false;
        self.sum = 0.0;
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            self.reset();
            return None;
        }
        if !self.started {
            self.started = true;
            self.sum = value;
        } else {
            self.sum += value;
        }
        Some(self.sum)
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        0
    }
}

#[inline]
pub fn advance_decline_line(
    input: &AdvanceDeclineLineInput,
) -> Result<AdvanceDeclineLineOutput, AdvanceDeclineLineError> {
    advance_decline_line_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn first_valid_value(data: &[f64]) -> usize {
    let mut i = 0usize;
    while i < data.len() {
        if data[i].is_finite() {
            break;
        }
        i += 1;
    }
    i.min(data.len())
}

#[inline(always)]
fn advance_decline_line_row(data: &[f64], out: &mut [f64]) {
    let mut started = false;
    let mut sum = 0.0;
    for (dst, &value) in out.iter_mut().zip(data.iter()) {
        if !value.is_finite() {
            *dst = f64::NAN;
            started = false;
            sum = 0.0;
            continue;
        }
        if !started {
            started = true;
            sum = value;
        } else {
            sum += value;
        }
        *dst = sum;
    }
}

#[inline(always)]
fn advance_decline_line_prepare<'a>(
    input: &'a AdvanceDeclineLineInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, Kernel), AdvanceDeclineLineError> {
    let data = input.as_ref();
    if data.is_empty() {
        return Err(AdvanceDeclineLineError::EmptyInputData);
    }

    let first = first_valid_value(data);
    if first >= data.len() {
        return Err(AdvanceDeclineLineError::AllValuesNaN);
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other.to_non_batch(),
    };
    Ok((data, first, chosen))
}

#[inline]
pub fn advance_decline_line_with_kernel(
    input: &AdvanceDeclineLineInput,
    kernel: Kernel,
) -> Result<AdvanceDeclineLineOutput, AdvanceDeclineLineError> {
    let (data, first, _chosen) = advance_decline_line_prepare(input, kernel)?;
    let mut values = alloc_with_nan_prefix(data.len(), first);
    advance_decline_line_row(data, &mut values);
    Ok(AdvanceDeclineLineOutput { values })
}

#[inline]
pub fn advance_decline_line_into_slice(
    dst: &mut [f64],
    input: &AdvanceDeclineLineInput,
    kernel: Kernel,
) -> Result<(), AdvanceDeclineLineError> {
    let (data, _first, _chosen) = advance_decline_line_prepare(input, kernel)?;
    if dst.len() != data.len() {
        return Err(AdvanceDeclineLineError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }
    advance_decline_line_row(data, dst);
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn advance_decline_line_into(
    input: &AdvanceDeclineLineInput,
    out: &mut [f64],
) -> Result<(), AdvanceDeclineLineError> {
    advance_decline_line_into_slice(out, input, Kernel::Auto)
}

#[derive(Clone, Debug, Default)]
pub struct AdvanceDeclineLineBatchRange;

#[derive(Clone, Debug, Default)]
pub struct AdvanceDeclineLineBatchBuilder {
    kernel: Kernel,
}

impl AdvanceDeclineLineBatchBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<AdvanceDeclineLineBatchOutput, AdvanceDeclineLineError> {
        advance_decline_line_batch_with_kernel(data, &AdvanceDeclineLineBatchRange, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<AdvanceDeclineLineBatchOutput, AdvanceDeclineLineError> {
        self.apply_slice(advance_decline_line_source(candles, source))
    }

    #[inline]
    pub fn with_default_candles(
        candles: &Candles,
    ) -> Result<AdvanceDeclineLineBatchOutput, AdvanceDeclineLineError> {
        AdvanceDeclineLineBatchBuilder::new().apply_candles(candles, "close")
    }
}

#[derive(Clone, Debug)]
pub struct AdvanceDeclineLineBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<AdvanceDeclineLineParams>,
    pub rows: usize,
    pub cols: usize,
}

impl AdvanceDeclineLineBatchOutput {
    pub fn row_for_params(&self, _params: &AdvanceDeclineLineParams) -> Option<usize> {
        if self.rows == 0 {
            None
        } else {
            Some(0)
        }
    }

    pub fn values_for(&self, _params: &AdvanceDeclineLineParams) -> Option<&[f64]> {
        if self.rows == 0 {
            None
        } else {
            self.values.get(0..self.cols)
        }
    }
}

#[inline(always)]
fn expand_grid_advance_decline_line(
    range: &AdvanceDeclineLineBatchRange,
) -> Result<Vec<AdvanceDeclineLineParams>, AdvanceDeclineLineError> {
    let _ = range;
    Ok(vec![AdvanceDeclineLineParams])
}

#[inline]
pub fn advance_decline_line_batch_with_kernel(
    data: &[f64],
    sweep: &AdvanceDeclineLineBatchRange,
    kernel: Kernel,
) -> Result<AdvanceDeclineLineBatchOutput, AdvanceDeclineLineError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(AdvanceDeclineLineError::InvalidKernelForBatch(other)),
    };
    advance_decline_line_batch_par_slice(data, sweep, batch_kernel.to_non_batch())
}

#[inline]
pub fn advance_decline_line_batch_slice(
    data: &[f64],
    sweep: &AdvanceDeclineLineBatchRange,
    kernel: Kernel,
) -> Result<AdvanceDeclineLineBatchOutput, AdvanceDeclineLineError> {
    advance_decline_line_batch_inner(data, sweep, kernel, false)
}

#[inline]
pub fn advance_decline_line_batch_par_slice(
    data: &[f64],
    sweep: &AdvanceDeclineLineBatchRange,
    kernel: Kernel,
) -> Result<AdvanceDeclineLineBatchOutput, AdvanceDeclineLineError> {
    advance_decline_line_batch_inner(data, sweep, kernel, true)
}

#[inline(always)]
fn advance_decline_line_batch_inner(
    data: &[f64],
    sweep: &AdvanceDeclineLineBatchRange,
    _kernel: Kernel,
    _parallel: bool,
) -> Result<AdvanceDeclineLineBatchOutput, AdvanceDeclineLineError> {
    if data.is_empty() {
        return Err(AdvanceDeclineLineError::EmptyInputData);
    }
    let first = first_valid_value(data);
    if first >= data.len() {
        return Err(AdvanceDeclineLineError::AllValuesNaN);
    }

    let combos = expand_grid_advance_decline_line(sweep)?;
    let rows = combos.len();
    let cols = data.len();

    let mut buf_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut buf_mu, cols, &[first]);
    let mut guard = ManuallyDrop::new(buf_mu);
    let out =
        unsafe { std::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };
    advance_decline_line_row(data, &mut out[..cols]);

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(AdvanceDeclineLineBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn advance_decline_line_batch_inner_into(
    data: &[f64],
    sweep: &AdvanceDeclineLineBatchRange,
    _kernel: Kernel,
    _parallel: bool,
    out: &mut [f64],
) -> Result<Vec<AdvanceDeclineLineParams>, AdvanceDeclineLineError> {
    if data.is_empty() {
        return Err(AdvanceDeclineLineError::EmptyInputData);
    }
    let first = first_valid_value(data);
    if first >= data.len() {
        return Err(AdvanceDeclineLineError::AllValuesNaN);
    }
    let combos = expand_grid_advance_decline_line(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    let total =
        rows.checked_mul(cols)
            .ok_or_else(|| AdvanceDeclineLineError::OutputLengthMismatch {
                expected: usize::MAX,
                got: out.len(),
            })?;
    if out.len() != total {
        return Err(AdvanceDeclineLineError::OutputLengthMismatch {
            expected: total,
            got: out.len(),
        });
    }
    out[..first].fill(f64::NAN);
    advance_decline_line_row(data, &mut out[..cols]);
    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "advance_decline_line")]
#[pyo3(signature = (data, kernel=None))]
pub fn advance_decline_line_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data_slice: &[f64];
    let owned;
    data_slice = if let Ok(slice) = data.as_slice() {
        slice
    } else {
        owned = data.to_owned_array();
        owned.as_slice().unwrap()
    };
    let kern = validate_kernel(kernel, false)?;
    let input = AdvanceDeclineLineInput::from_slice(data_slice, AdvanceDeclineLineParams);
    let values = py
        .allow_threads(|| advance_decline_line_with_kernel(&input, kern).map(|out| out.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "AdvanceDeclineLineStream")]
pub struct AdvanceDeclineLineStreamPy {
    stream: AdvanceDeclineLineStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AdvanceDeclineLineStreamPy {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self {
            stream: AdvanceDeclineLineStream::try_new()
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "advance_decline_line_batch")]
#[pyo3(signature = (data, kernel=None))]
pub fn advance_decline_line_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data_slice: &[f64];
    let owned;
    data_slice = if let Ok(slice) = data.as_slice() {
        slice
    } else {
        owned = data.to_owned_array();
        owned.as_slice().unwrap()
    };
    let kern = validate_kernel(kernel, true)?;
    if data_slice.is_empty() {
        return Err(PyValueError::new_err(
            AdvanceDeclineLineError::EmptyInputData.to_string(),
        ));
    }

    let rows = 1usize;
    let cols = data_slice.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("advance_decline_line_batch: size overflow"))?;
    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    py.allow_threads(|| {
        let kernel = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        advance_decline_line_batch_inner_into(
            data_slice,
            &AdvanceDeclineLineBatchRange,
            kernel,
            true,
            slice_out,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item("params", Vec::<f64>::new().into_pyarray(py))?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_advance_decline_line_module(
    module: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(advance_decline_line_py, module)?)?;
    module.add_function(wrap_pyfunction!(advance_decline_line_batch_py, module)?)?;
    module.add_class::<AdvanceDeclineLineStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "advance_decline_line_js")]
pub fn advance_decline_line_js(data: &[f64]) -> Result<Vec<f64>, JsValue> {
    let input = AdvanceDeclineLineInput::from_slice(data, AdvanceDeclineLineParams);
    advance_decline_line_with_kernel(&input, Kernel::Auto)
        .map(|out| out.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn advance_decline_line_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn advance_decline_line_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn advance_decline_line_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        let input = AdvanceDeclineLineInput::from_slice(data, AdvanceDeclineLineParams);
        advance_decline_line_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct AdvanceDeclineLineBatchConfig {}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct AdvanceDeclineLineBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<AdvanceDeclineLineParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "advance_decline_line_batch_js")]
pub fn advance_decline_line_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let _: AdvanceDeclineLineBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let output =
        advance_decline_line_batch_with_kernel(data, &AdvanceDeclineLineBatchRange, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&AdvanceDeclineLineBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn advance_decline_line_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        advance_decline_line_batch_inner_into(
            data,
            &AdvanceDeclineLineBatchRange,
            Kernel::Auto,
            false,
            out,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(1)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn advance_decline_line_output_into_js(
    data: &[f64],
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values = advance_decline_line_js(data)?;
    crate::write_wasm_f64_output("advance_decline_line_output_into_js", &values, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn advance_decline_line_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = advance_decline_line_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "advance_decline_line_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use std::error::Error;

    fn load_close() -> Result<Vec<f64>, Box<dyn Error>> {
        let candles = read_candles_from_csv("src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv")?;
        Ok(candles.close)
    }

    #[test]
    fn advance_decline_line_basic_slice() -> Result<(), Box<dyn Error>> {
        let data = [1.0, 2.0, 3.0, 4.0];
        let input = AdvanceDeclineLineInput::from_slice(&data, AdvanceDeclineLineParams);
        let out = advance_decline_line_with_kernel(&input, Kernel::Scalar)?;
        assert_eq!(out.values, vec![1.0, 3.0, 6.0, 10.0]);
        Ok(())
    }

    #[test]
    fn advance_decline_line_nan_resets_segment() -> Result<(), Box<dyn Error>> {
        let data = [f64::NAN, 1.0, 2.0, f64::NAN, 3.0, 4.0];
        let input = AdvanceDeclineLineInput::from_slice(&data, AdvanceDeclineLineParams);
        let out = advance_decline_line_with_kernel(&input, Kernel::Scalar)?;
        assert!(out.values[0].is_nan());
        assert_eq!(out.values[1], 1.0);
        assert_eq!(out.values[2], 3.0);
        assert!(out.values[3].is_nan());
        assert_eq!(out.values[4], 3.0);
        assert_eq!(out.values[5], 7.0);
        Ok(())
    }

    #[test]
    fn advance_decline_line_output_contract() -> Result<(), Box<dyn Error>> {
        let close = load_close()?;
        let input = AdvanceDeclineLineInput::from_slice(&close, AdvanceDeclineLineParams);
        let out = advance_decline_line_with_kernel(&input, Kernel::Scalar)?;
        assert_eq!(out.values.len(), close.len());
        assert!(out.values.iter().all(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn advance_decline_line_auto_matches_scalar() -> Result<(), Box<dyn Error>> {
        let close = load_close()?;
        let input = AdvanceDeclineLineInput::from_slice(&close, AdvanceDeclineLineParams);
        let auto = advance_decline_line_with_kernel(&input, Kernel::Auto)?;
        let scalar = advance_decline_line_with_kernel(&input, Kernel::Scalar)?;
        assert_eq!(auto.values, scalar.values);
        Ok(())
    }

    #[test]
    fn advance_decline_line_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let close = load_close()?;
        let input = AdvanceDeclineLineInput::from_slice(&close, AdvanceDeclineLineParams);
        let batch = advance_decline_line_with_kernel(&input, Kernel::Scalar)?;
        let mut stream = AdvanceDeclineLineStream::try_new()?;
        let mut streamed = Vec::with_capacity(close.len());
        for &value in &close {
            streamed.push(stream.update(value).unwrap_or(f64::NAN));
        }
        assert_eq!(streamed, batch.values);
        Ok(())
    }

    #[test]
    fn advance_decline_line_batch_matches_single() -> Result<(), Box<dyn Error>> {
        let close = load_close()?;
        let batch = advance_decline_line_batch_with_kernel(
            &close,
            &AdvanceDeclineLineBatchRange,
            Kernel::ScalarBatch,
        )?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        let single = advance_decline_line_with_kernel(
            &AdvanceDeclineLineInput::from_slice(&close, AdvanceDeclineLineParams),
            Kernel::Scalar,
        )?;
        assert_eq!(batch.values, single.values);
        Ok(())
    }
}
