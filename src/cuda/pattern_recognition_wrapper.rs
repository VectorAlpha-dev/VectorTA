#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{CopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaPatternRecognitionError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
}

pub struct DevicePatternFeatures {
    pub body: DeviceBuffer<f32>,
    pub body_low: DeviceBuffer<f32>,
    pub body_high: DeviceBuffer<f32>,
    pub range: DeviceBuffer<f32>,
    pub upper_shadow: DeviceBuffer<f32>,
    pub lower_shadow: DeviceBuffer<f32>,
    pub direction: DeviceBuffer<i8>,
    pub body_gap_up: DeviceBuffer<u8>,
    pub body_gap_down: DeviceBuffer<u8>,
}

impl DevicePatternFeatures {
    pub fn len(&self) -> usize {
        self.body.len()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NativeSubsetRows {
    pub cdldoji: usize,
    pub cdldragonflydoji: usize,
    pub cdlgravestonedoji: usize,
    pub cdllongleggeddoji: usize,
    pub cdlmarubozu: usize,
}

const NATIVE_SUPPORTED_PATTERN_IDS: [&str; 61] = [
    "cdl2crows",
    "cdl3blackcrows",
    "cdl3inside",
    "cdl3linestrike",
    "cdl3outside",
    "cdl3starsinsouth",
    "cdl3whitesoldiers",
    "cdlabandonedbaby",
    "cdladvanceblock",
    "cdlbelthold",
    "cdlbreakaway",
    "cdlclosingmarubozu",
    "cdlconcealbabyswall",
    "cdlcounterattack",
    "cdldarkcloudcover",
    "cdldoji",
    "cdldojistar",
    "cdldragonflydoji",
    "cdlengulfing",
    "cdleveningdojistar",
    "cdleveningstar",
    "cdlmorningstar",
    "cdlgravestonedoji",
    "cdlhammer",
    "cdlhangingman",
    "cdlharami",
    "cdlharamicross",
    "cdlhighwave",
    "cdlinvertedhammer",
    "cdllongleggeddoji",
    "cdllongline",
    "cdlmarubozu",
    "cdlrickshawman",
    "cdlshootingstar",
    "cdlshortline",
    "cdlspinningtop",
    "cdltakuri",
    "cdlhomingpigeon",
    "cdlmatchinglow",
    "cdlinneck",
    "cdlonneck",
    "cdlpiercing",
    "cdlthrusting",
    "cdlmorningdojistar",
    "cdltristar",
    "cdlidentical3crows",
    "cdlsticksandwich",
    "cdlseparatinglines",
    "cdlgapsidesidewhite",
    "cdlhikkake",
    "cdlhikkakemod",
    "cdlkicking",
    "cdlkickingbylength",
    "cdlladderbottom",
    "cdlmathold",
    "cdlrisefall3methods",
    "cdlstalledpattern",
    "cdltasukigap",
    "cdlunique3river",
    "cdlupsidegap2crows",
    "cdlxsidegap3methods",
];

pub struct CudaPatternRecognition {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaPatternRecognition {
    pub fn new(device_id: usize) -> Result<Self, CudaPatternRecognitionError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(
            env!("OUT_DIR"),
            "/pattern_recognition_kernel.ptx"
        ));
        let module = Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O4),
            ],
        )
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

    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    pub fn context_arc(&self) -> Arc<Context> {
        self.context.clone()
    }

    pub fn synchronize(&self) -> Result<(), CudaPatternRecognitionError> {
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn allocate_feature_buffers(
        &self,
        len: usize,
    ) -> Result<DevicePatternFeatures, CudaPatternRecognitionError> {
        if len == 0 {
            return Err(CudaPatternRecognitionError::InvalidInput(
                "len must be > 0".to_string(),
            ));
        }

        let body = unsafe { DeviceBuffer::<f32>::uninitialized(len) }?;
        let body_low = unsafe { DeviceBuffer::<f32>::uninitialized(len) }?;
        let body_high = unsafe { DeviceBuffer::<f32>::uninitialized(len) }?;
        let range = unsafe { DeviceBuffer::<f32>::uninitialized(len) }?;
        let upper_shadow = unsafe { DeviceBuffer::<f32>::uninitialized(len) }?;
        let lower_shadow = unsafe { DeviceBuffer::<f32>::uninitialized(len) }?;
        let direction = unsafe { DeviceBuffer::<i8>::uninitialized(len) }?;
        let body_gap_up = unsafe { DeviceBuffer::<u8>::uninitialized(len) }?;
        let body_gap_down = unsafe { DeviceBuffer::<u8>::uninitialized(len) }?;

        Ok(DevicePatternFeatures {
            body,
            body_low,
            body_high,
            range,
            upper_shadow,
            lower_shadow,
            direction,
            body_gap_up,
            body_gap_down,
        })
    }

    pub fn compute_features_device(
        &self,
        open: &[f32],
        high: &[f32],
        low: &[f32],
        close: &[f32],
    ) -> Result<DevicePatternFeatures, CudaPatternRecognitionError> {
        let len = validate_ohlc_len(open, high, low, close)?;

        let d_open = DeviceBuffer::from_slice(open)?;
        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;

        let mut out = self.allocate_feature_buffers(len)?;
        self.compute_features_device_into(&d_open, &d_high, &d_low, &d_close, len, &mut out)?;
        self.synchronize()?;

        Ok(out)
    }

    pub fn compute_features_device_into(
        &self,
        d_open: &DeviceBuffer<f32>,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        len: usize,
        out: &mut DevicePatternFeatures,
    ) -> Result<(), CudaPatternRecognitionError> {
        if len == 0 {
            return Err(CudaPatternRecognitionError::InvalidInput(
                "len must be > 0".to_string(),
            ));
        }

        if d_open.len() < len || d_high.len() < len || d_low.len() < len || d_close.len() < len {
            return Err(CudaPatternRecognitionError::InvalidInput(
                "input buffer too small for len".to_string(),
            ));
        }

        if out.len() < len {
            return Err(CudaPatternRecognitionError::InvalidInput(
                "output buffer too small for len".to_string(),
            ));
        }

        let func = self
            .module
            .get_function("pattern_features_kernel_f32")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_features_kernel_f32",
            })?;

        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut open_ptr = d_open.as_device_ptr().as_raw();
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut body_ptr = out.body.as_device_ptr().as_raw();
            let mut body_low_ptr = out.body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = out.body_high.as_device_ptr().as_raw();
            let mut range_ptr = out.range.as_device_ptr().as_raw();
            let mut upper_ptr = out.upper_shadow.as_device_ptr().as_raw();
            let mut lower_ptr = out.lower_shadow.as_device_ptr().as_raw();
            let mut dir_ptr = out.direction.as_device_ptr().as_raw();
            let mut gap_up_ptr = out.body_gap_up.as_device_ptr().as_raw();
            let mut gap_down_ptr = out.body_gap_down.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut open_ptr as *mut _ as *mut c_void,
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut range_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut gap_up_ptr as *mut _ as *mut c_void,
                &mut gap_down_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    pub fn doji_mask_from_features_device_into(
        &self,
        body: &DeviceBuffer<f32>,
        range: &DeviceBuffer<f32>,
        len: usize,
        threshold: f32,
        out_mask: &mut DeviceBuffer<u8>,
    ) -> Result<(), CudaPatternRecognitionError> {
        if len == 0 {
            return Err(CudaPatternRecognitionError::InvalidInput(
                "len must be > 0".to_string(),
            ));
        }

        if body.len() < len || range.len() < len || out_mask.len() < len {
            return Err(CudaPatternRecognitionError::InvalidInput(
                "buffer too small for len".to_string(),
            ));
        }

        let func = self
            .module
            .get_function("pattern_doji_predicate_kernel_f32")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_doji_predicate_kernel_f32",
            })?;

        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut range_ptr = range.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut threshold_f = threshold;
            let mut out_ptr = out_mask.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut range_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut threshold_f as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    pub fn doji_mask_from_features_host(
        &self,
        body: &[f32],
        range: &[f32],
        threshold: f32,
    ) -> Result<Vec<u8>, CudaPatternRecognitionError> {
        if body.len() != range.len() || body.is_empty() {
            return Err(CudaPatternRecognitionError::InvalidInput(
                "body/range must be non-empty and same length".to_string(),
            ));
        }

        let len = body.len();
        let d_body = DeviceBuffer::from_slice(body)?;
        let d_range = DeviceBuffer::from_slice(range)?;
        let mut d_out = unsafe { DeviceBuffer::<u8>::uninitialized(len) }?;

        self.doji_mask_from_features_device_into(&d_body, &d_range, len, threshold, &mut d_out)?;
        self.synchronize()?;

        let mut host = vec![0u8; len];
        d_out.copy_to(&mut host)?;
        Ok(host)
    }

    pub fn native_supported_pattern_ids() -> &'static [&'static str] {
        &NATIVE_SUPPORTED_PATTERN_IDS
    }

    pub fn compute_native_matrix_device(
        &self,
        features: &DevicePatternFeatures,
        rows: usize,
        cols: usize,
        row_map: &[(&str, usize)],
    ) -> Result<DeviceBuffer<u8>, CudaPatternRecognitionError> {
        if rows == 0 || cols == 0 {
            return Err(CudaPatternRecognitionError::InvalidInput(
                "rows and cols must be > 0".to_string(),
            ));
        }

        if features.len() != cols {
            return Err(CudaPatternRecognitionError::InvalidInput(format!(
                "features length mismatch: features={} cols={}",
                features.len(),
                cols
            )));
        }

        for (_, row) in row_map {
            if *row >= rows {
                return Err(CudaPatternRecognitionError::InvalidInput(
                    "row index out of bounds".to_string(),
                ));
            }
        }

        let total = rows.checked_mul(cols).ok_or_else(|| {
            CudaPatternRecognitionError::InvalidInput("rows*cols overflow".to_string())
        })?;
        let mut d_matrix = DeviceBuffer::<u8>::zeroed(total)?;

        for (pattern_id, row) in row_map {
            self.launch_pattern_row(features, cols, &mut d_matrix, cols, *row, pattern_id)?;
        }

        Ok(d_matrix)
    }

    pub fn compute_native_matrix_host(
        &self,
        features: &DevicePatternFeatures,
        rows: usize,
        cols: usize,
        row_map: &[(&str, usize)],
    ) -> Result<Vec<u8>, CudaPatternRecognitionError> {
        let d_matrix = self.compute_native_matrix_device(features, rows, cols, row_map)?;
        self.synchronize()?;
        let mut host = vec![0u8; rows.saturating_mul(cols)];
        d_matrix.copy_to(&mut host)?;
        Ok(host)
    }

    pub fn compute_native_subset_matrix_device(
        &self,
        features: &DevicePatternFeatures,
        rows: usize,
        cols: usize,
        subset_rows: NativeSubsetRows,
    ) -> Result<DeviceBuffer<u8>, CudaPatternRecognitionError> {
        let row_map = [
            ("cdldoji", subset_rows.cdldoji),
            ("cdldragonflydoji", subset_rows.cdldragonflydoji),
            ("cdlgravestonedoji", subset_rows.cdlgravestonedoji),
            ("cdllongleggeddoji", subset_rows.cdllongleggeddoji),
            ("cdlmarubozu", subset_rows.cdlmarubozu),
        ];
        self.compute_native_matrix_device(features, rows, cols, &row_map)
    }

    pub fn compute_native_subset_matrix_host(
        &self,
        features: &DevicePatternFeatures,
        rows: usize,
        cols: usize,
        subset_rows: NativeSubsetRows,
    ) -> Result<Vec<u8>, CudaPatternRecognitionError> {
        let row_map = [
            ("cdldoji", subset_rows.cdldoji),
            ("cdldragonflydoji", subset_rows.cdldragonflydoji),
            ("cdlgravestonedoji", subset_rows.cdlgravestonedoji),
            ("cdllongleggeddoji", subset_rows.cdllongleggeddoji),
            ("cdlmarubozu", subset_rows.cdlmarubozu),
        ];
        self.compute_native_matrix_host(features, rows, cols, &row_map)
    }

    fn launch_pattern_row(
        &self,
        features: &DevicePatternFeatures,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
        pattern_id: &str,
    ) -> Result<(), CudaPatternRecognitionError> {
        match pattern_id {
            "cdlbelthold" => self.launch_row_cdlbelthold(
                &features.body,
                &features.upper_shadow,
                &features.lower_shadow,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlclosingmarubozu" => self.launch_row_cdlclosingmarubozu(
                &features.body,
                &features.upper_shadow,
                &features.lower_shadow,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlcounterattack" => self.launch_row_cdlcounterattack(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdleveningdojistar" => self.launch_row_cdleveningdojistar(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.direction,
                &features.body_gap_up,
                len,
                matrix,
                cols,
                row,
            ),
            "cdleveningstar" => self.launch_row_cdleveningstar(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.direction,
                &features.body_gap_up,
                len,
                matrix,
                cols,
                row,
            ),
            "cdldoji" => self.launch_row_cdldoji(&features.body, len, matrix, cols, row),
            "cdldojistar" => self.launch_row_cdldojistar(
                &features.body,
                &features.direction,
                &features.body_gap_up,
                &features.body_gap_down,
                len,
                matrix,
                cols,
                row,
            ),
            "cdldarkcloudcover" => self.launch_row_cdldarkcloudcover(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.upper_shadow,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdldragonflydoji" => self.launch_row_cdldragonflydoji(
                &features.body,
                &features.upper_shadow,
                &features.lower_shadow,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlengulfing" => self.launch_row_cdlengulfing(
                &features.body_low,
                &features.body_high,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlgapsidesidewhite" => self.launch_row_cdlgapsidesidewhite(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.direction,
                &features.body_gap_up,
                &features.body_gap_down,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlgravestonedoji" => self.launch_row_cdlgravestonedoji(
                &features.body,
                &features.upper_shadow,
                &features.lower_shadow,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlhammer" => self.launch_row_cdlhammer(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.upper_shadow,
                &features.lower_shadow,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlhangingman" => self.launch_row_cdlhangingman(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.upper_shadow,
                &features.lower_shadow,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlharami" => self.launch_row_cdlharami(
                &features.body,
                &features.body_low,
                &features.body_high,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlharamicross" => self.launch_row_cdlharami(
                &features.body,
                &features.body_low,
                &features.body_high,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlhikkake" => self.launch_row_cdlhikkake(
                &features.body_low,
                &features.body_high,
                &features.upper_shadow,
                &features.lower_shadow,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlhikkakemod" => self.launch_row_cdlhikkakemod(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.upper_shadow,
                &features.lower_shadow,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlhighwave" => self.launch_row_cdlhighwave(
                &features.body,
                &features.upper_shadow,
                &features.lower_shadow,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlhomingpigeon" => self.launch_row_cdlhomingpigeon(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlinneck" => self.launch_row_cdlinneck(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.lower_shadow,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlinvertedhammer" => self.launch_row_cdlinvertedhammer(
                &features.body,
                &features.upper_shadow,
                &features.lower_shadow,
                &features.body_gap_down,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlladderbottom" => self.launch_row_cdlladderbottom(
                &features.body_low,
                &features.body_high,
                &features.upper_shadow,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdllongleggeddoji" => self.launch_row_cdllongleggeddoji(
                &features.body,
                &features.upper_shadow,
                &features.lower_shadow,
                len,
                matrix,
                cols,
                row,
            ),
            "cdllongline" => self.launch_row_cdllongline(
                &features.body,
                &features.upper_shadow,
                &features.lower_shadow,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlmarubozu" => self.launch_row_cdlmarubozu(
                &features.body,
                &features.upper_shadow,
                &features.lower_shadow,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlmatchinglow" => self.launch_row_cdlmatchinglow(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlmorningdojistar" => self.launch_row_cdlmorningdojistar(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.direction,
                &features.body_gap_down,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlmorningstar" => self.launch_row_cdlmorningstar(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.direction,
                &features.body_gap_down,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlonneck" => self.launch_row_cdlonneck(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.lower_shadow,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlpiercing" => self.launch_row_cdlpiercing(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.lower_shadow,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlrickshawman" => self.launch_row_cdlrickshawman(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.upper_shadow,
                &features.lower_shadow,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlseparatinglines" => self.launch_row_cdlseparatinglines(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.upper_shadow,
                &features.lower_shadow,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlshootingstar" => self.launch_row_cdlshootingstar(
                &features.body,
                &features.upper_shadow,
                &features.lower_shadow,
                &features.body_gap_up,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlshortline" => self.launch_row_cdlshortline(
                &features.body,
                &features.upper_shadow,
                &features.lower_shadow,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlspinningtop" => self.launch_row_cdlspinningtop(
                &features.body,
                &features.upper_shadow,
                &features.lower_shadow,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlstalledpattern" => self.launch_row_cdlstalledpattern(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.upper_shadow,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlsticksandwich" => self.launch_row_cdlsticksandwich(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.lower_shadow,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdltakuri" => self.launch_row_cdltakuri(
                &features.body,
                &features.upper_shadow,
                &features.lower_shadow,
                len,
                matrix,
                cols,
                row,
            ),
            "cdltasukigap" => self.launch_row_cdltasukigap(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.direction,
                &features.body_gap_up,
                &features.body_gap_down,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlthrusting" => self.launch_row_cdlthrusting(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.lower_shadow,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlkicking" => self.launch_row_cdlkicking(
                &features.body,
                &features.upper_shadow,
                &features.lower_shadow,
                &features.direction,
                &features.body_gap_up,
                &features.body_gap_down,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlkickingbylength" => self.launch_row_cdlkickingbylength(
                &features.body,
                &features.upper_shadow,
                &features.lower_shadow,
                &features.direction,
                &features.body_gap_up,
                &features.body_gap_down,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlidentical3crows" => self.launch_row_cdlidentical3crows(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.lower_shadow,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdltristar" => self.launch_row_cdltristar(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.body_gap_up,
                &features.body_gap_down,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlunique3river" => self.launch_row_cdlunique3river(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.lower_shadow,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlupsidegap2crows" => self.launch_row_cdlupsidegap2crows(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.direction,
                &features.body_gap_up,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlxsidegap3methods" => self.launch_row_cdlxsidegap3methods(
                &features.body_low,
                &features.body_high,
                &features.direction,
                &features.body_gap_up,
                &features.body_gap_down,
                len,
                matrix,
                cols,
                row,
            ),
            "cdl2crows" => self.launch_row_cdl2crows(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdl3blackcrows" => self.launch_row_cdl3blackcrows(
                &features.body_low,
                &features.body_high,
                &features.upper_shadow,
                &features.lower_shadow,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdl3inside" => self.launch_row_cdl3inside(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdl3linestrike" => self.launch_row_cdl3linestrike(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdl3outside" => self.launch_row_cdl3outside(
                &features.body_low,
                &features.body_high,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdl3starsinsouth" => self.launch_row_cdl3starsinsouth(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.upper_shadow,
                &features.lower_shadow,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdl3whitesoldiers" => self.launch_row_cdl3whitesoldiers(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.upper_shadow,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlabandonedbaby" => self.launch_row_cdlabandonedbaby(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.upper_shadow,
                &features.lower_shadow,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdladvanceblock" => self.launch_row_cdladvanceblock(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.upper_shadow,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlbreakaway" => self.launch_row_cdlbreakaway(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.upper_shadow,
                &features.lower_shadow,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlconcealbabyswall" => self.launch_row_cdlconcealbabyswall(
                &features.body_low,
                &features.body_high,
                &features.upper_shadow,
                &features.lower_shadow,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlmathold" => self.launch_row_cdlmathold(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.upper_shadow,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlrisefall3methods" => self.launch_row_cdlrisefall3methods(
                &features.body,
                &features.body_low,
                &features.body_high,
                &features.upper_shadow,
                &features.lower_shadow,
                &features.direction,
                len,
                matrix,
                cols,
                row,
            ),
            _ => Err(CudaPatternRecognitionError::InvalidInput(format!(
                "pattern not supported by native CUDA matrix: {pattern_id}"
            ))),
        }
    }

    fn launch_row_cdldoji(
        &self,
        body: &DeviceBuffer<f32>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdldoji_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdldoji_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut period_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdldragonflydoji(
        &self,
        body: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdldragonflydoji_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdldragonflydoji_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut period_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlgravestonedoji(
        &self,
        body: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlgravestonedoji_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlgravestonedoji_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut period_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdllongleggeddoji(
        &self,
        body: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdllongleggeddoji_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdllongleggeddoji_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut period_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlmarubozu(
        &self,
        body: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlmarubozu_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlmarubozu_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut period_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlbelthold(
        &self,
        body: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlbelthold_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlbelthold_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut body_period_i = 10i32;
            let mut shadow_period_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut body_period_i as *mut _ as *mut c_void,
                &mut shadow_period_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlclosingmarubozu(
        &self,
        body: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlclosingmarubozu_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlclosingmarubozu_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut body_period_i = 10i32;
            let mut shadow_period_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut body_period_i as *mut _ as *mut c_void,
                &mut shadow_period_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlhammer(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlhammer_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlhammer_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut body_period_i = 10i32;
            let mut shadow_long_i = 10i32;
            let mut shadow_short_i = 10i32;
            let mut near_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut body_period_i as *mut _ as *mut c_void,
                &mut shadow_long_i as *mut _ as *mut c_void,
                &mut shadow_short_i as *mut _ as *mut c_void,
                &mut near_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlhangingman(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlhangingman_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlhangingman_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut body_period_i = 10i32;
            let mut shadow_long_i = 10i32;
            let mut shadow_short_i = 10i32;
            let mut near_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut body_period_i as *mut _ as *mut c_void,
                &mut shadow_long_i as *mut _ as *mut c_void,
                &mut shadow_short_i as *mut _ as *mut c_void,
                &mut near_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlrickshawman(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlrickshawman_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlrickshawman_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut body_period_i = 10i32;
            let mut shadow_long_i = 10i32;
            let mut near_i = 5i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut body_period_i as *mut _ as *mut c_void,
                &mut shadow_long_i as *mut _ as *mut c_void,
                &mut near_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlmatchinglow(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlmatchinglow_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlmatchinglow_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut period_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlinneck(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlinneck_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlinneck_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut equal_i = 10i32;
            let mut long_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut equal_i as *mut _ as *mut c_void,
                &mut long_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlonneck(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlonneck_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlonneck_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut equal_i = 10i32;
            let mut long_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut equal_i as *mut _ as *mut c_void,
                &mut long_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlpiercing(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlpiercing_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlpiercing_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut long_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut long_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlthrusting(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlthrusting_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlthrusting_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut equal_i = 10i32;
            let mut long_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut equal_i as *mut _ as *mut c_void,
                &mut long_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdleveningdojistar(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        body_gap_up: &DeviceBuffer<u8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdleveningdojistar_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdleveningdojistar_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut gap_ptr = body_gap_up.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut period_long_i = 10i32;
            let mut period_doji_i = 10i32;
            let mut period_short_i = 10i32;
            let mut penetration = 0.3f32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut gap_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut period_long_i as *mut _ as *mut c_void,
                &mut period_doji_i as *mut _ as *mut c_void,
                &mut period_short_i as *mut _ as *mut c_void,
                &mut penetration as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdleveningstar(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        body_gap_up: &DeviceBuffer<u8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdleveningstar_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdleveningstar_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut gap_ptr = body_gap_up.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut period_long_i = 10i32;
            let mut period_short1_i = 10i32;
            let mut period_short0_i = 10i32;
            let mut penetration = 0.3f32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut gap_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut period_long_i as *mut _ as *mut c_void,
                &mut period_short1_i as *mut _ as *mut c_void,
                &mut period_short0_i as *mut _ as *mut c_void,
                &mut penetration as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlmorningdojistar(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        body_gap_down: &DeviceBuffer<u8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlmorningdojistar_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlmorningdojistar_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut gap_ptr = body_gap_down.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut period_long_i = 10i32;
            let mut period_doji_i = 10i32;
            let mut period_short_i = 10i32;
            let mut penetration = 0.3f32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut gap_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut period_long_i as *mut _ as *mut c_void,
                &mut period_doji_i as *mut _ as *mut c_void,
                &mut period_short_i as *mut _ as *mut c_void,
                &mut penetration as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlmorningstar(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        body_gap_down: &DeviceBuffer<u8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlmorningstar_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlmorningstar_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut gap_ptr = body_gap_down.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut period_long_i = 10i32;
            let mut period_short1_i = 10i32;
            let mut period_short0_i = 10i32;
            let mut penetration = 0.3f32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut gap_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut period_long_i as *mut _ as *mut c_void,
                &mut period_short1_i as *mut _ as *mut c_void,
                &mut period_short0_i as *mut _ as *mut c_void,
                &mut penetration as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlgapsidesidewhite(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        body_gap_up: &DeviceBuffer<u8>,
        body_gap_down: &DeviceBuffer<u8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlgapsidesidewhite_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlgapsidesidewhite_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut gap_up_ptr = body_gap_up.as_device_ptr().as_raw();
            let mut gap_down_ptr = body_gap_down.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut near_i = 10i32;
            let mut equal_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut gap_up_ptr as *mut _ as *mut c_void,
                &mut gap_down_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut near_i as *mut _ as *mut c_void,
                &mut equal_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlkicking(
        &self,
        body: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        body_gap_up: &DeviceBuffer<u8>,
        body_gap_down: &DeviceBuffer<u8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlkicking_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlkicking_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut gap_up_ptr = body_gap_up.as_device_ptr().as_raw();
            let mut gap_down_ptr = body_gap_down.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut shadow_i = 10i32;
            let mut body_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut gap_up_ptr as *mut _ as *mut c_void,
                &mut gap_down_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut shadow_i as *mut _ as *mut c_void,
                &mut body_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlkickingbylength(
        &self,
        body: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        body_gap_up: &DeviceBuffer<u8>,
        body_gap_down: &DeviceBuffer<u8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlkickingbylength_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlkickingbylength_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut gap_up_ptr = body_gap_up.as_device_ptr().as_raw();
            let mut gap_down_ptr = body_gap_down.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut shadow_i = 10i32;
            let mut body_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut gap_up_ptr as *mut _ as *mut c_void,
                &mut gap_down_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut shadow_i as *mut _ as *mut c_void,
                &mut body_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlidentical3crows(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlidentical3crows_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlidentical3crows_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut shadow_i = 10i32;
            let mut equal_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut shadow_i as *mut _ as *mut c_void,
                &mut equal_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlsticksandwich(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlsticksandwich_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlsticksandwich_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut equal_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut equal_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlseparatinglines(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlseparatinglines_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlseparatinglines_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut shadow_i = 10i32;
            let mut body_long_i = 10i32;
            let mut equal_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut shadow_i as *mut _ as *mut c_void,
                &mut body_long_i as *mut _ as *mut c_void,
                &mut equal_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlcounterattack(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlcounterattack_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlcounterattack_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut equal_i = 10i32;
            let mut body_long_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut equal_i as *mut _ as *mut c_void,
                &mut body_long_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    fn launch_row_cdldarkcloudcover(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdldarkcloudcover_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdldarkcloudcover_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut body_long_i = 10i32;
            let mut penetration = 0.5f32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut body_long_i as *mut _ as *mut c_void,
                &mut penetration as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    fn launch_row_cdlxsidegap3methods(
        &self,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        body_gap_up: &DeviceBuffer<u8>,
        body_gap_down: &DeviceBuffer<u8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlxsidegap3methods_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlxsidegap3methods_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);
        unsafe {
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut gap_up_ptr = body_gap_up.as_device_ptr().as_raw();
            let mut gap_down_ptr = body_gap_down.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut gap_up_ptr as *mut _ as *mut c_void,
                &mut gap_down_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    fn launch_row_cdlupsidegap2crows(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        body_gap_up: &DeviceBuffer<u8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlupsidegap2crows_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlupsidegap2crows_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);
        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut gap_up_ptr = body_gap_up.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut short_i = 10i32;
            let mut long_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut gap_up_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut short_i as *mut _ as *mut c_void,
                &mut long_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    fn launch_row_cdlunique3river(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlunique3river_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlunique3river_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);
        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut short_i = 10i32;
            let mut long_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut short_i as *mut _ as *mut c_void,
                &mut long_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    fn launch_row_cdltasukigap(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        body_gap_up: &DeviceBuffer<u8>,
        body_gap_down: &DeviceBuffer<u8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdltasukigap_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdltasukigap_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);
        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut gap_up_ptr = body_gap_up.as_device_ptr().as_raw();
            let mut gap_down_ptr = body_gap_down.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut near_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut gap_up_ptr as *mut _ as *mut c_void,
                &mut gap_down_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut near_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    fn launch_row_cdlladderbottom(
        &self,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlladderbottom_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlladderbottom_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);
        unsafe {
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut shadow_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut shadow_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    fn launch_row_cdlstalledpattern(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlstalledpattern_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlstalledpattern_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);
        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut body_long_i = 10i32;
            let mut body_short_i = 10i32;
            let mut shadow_i = 10i32;
            let mut near_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut body_long_i as *mut _ as *mut c_void,
                &mut body_short_i as *mut _ as *mut c_void,
                &mut shadow_i as *mut _ as *mut c_void,
                &mut near_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    fn launch_row_cdlhikkake(
        &self,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlhikkake_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlhikkake_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);
        unsafe {
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    fn launch_row_cdlhikkakemod(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlhikkakemod_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlhikkakemod_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);
        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut near_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut near_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    fn launch_row_cdldojistar(
        &self,
        body: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        body_gap_up: &DeviceBuffer<u8>,
        body_gap_down: &DeviceBuffer<u8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdldojistar_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdldojistar_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut gap_up_ptr = body_gap_up.as_device_ptr().as_raw();
            let mut gap_down_ptr = body_gap_down.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut period_long_i = 10i32;
            let mut period_doji_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut gap_up_ptr as *mut _ as *mut c_void,
                &mut gap_down_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut period_long_i as *mut _ as *mut c_void,
                &mut period_doji_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlengulfing(
        &self,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlengulfing_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlengulfing_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlharami(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlharami_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlharami_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut period_long_i = 10i32;
            let mut period_short_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut period_long_i as *mut _ as *mut c_void,
                &mut period_short_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlhighwave(
        &self,
        body: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlhighwave_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlhighwave_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut period_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlhomingpigeon(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlhomingpigeon_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlhomingpigeon_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut period_long_i = 10i32;
            let mut period_short_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut period_long_i as *mut _ as *mut c_void,
                &mut period_short_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlinvertedhammer(
        &self,
        body: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        body_gap_down: &DeviceBuffer<u8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlinvertedhammer_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlinvertedhammer_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut gap_ptr = body_gap_down.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut period_body_i = 10i32;
            let mut period_upper_i = 10i32;
            let mut period_lower_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut gap_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut period_body_i as *mut _ as *mut c_void,
                &mut period_upper_i as *mut _ as *mut c_void,
                &mut period_lower_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdllongline(
        &self,
        body: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdllongline_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdllongline_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut body_period_i = 10i32;
            let mut shadow_period_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut body_period_i as *mut _ as *mut c_void,
                &mut shadow_period_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlshootingstar(
        &self,
        body: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        body_gap_up: &DeviceBuffer<u8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlshootingstar_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlshootingstar_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut gap_ptr = body_gap_up.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut period_body_i = 10i32;
            let mut period_upper_i = 10i32;
            let mut period_lower_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut gap_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut period_body_i as *mut _ as *mut c_void,
                &mut period_upper_i as *mut _ as *mut c_void,
                &mut period_lower_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlshortline(
        &self,
        body: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlshortline_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlshortline_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut body_period_i = 10i32;
            let mut shadow_period_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut body_period_i as *mut _ as *mut c_void,
                &mut shadow_period_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlspinningtop(
        &self,
        body: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlspinningtop_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlspinningtop_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut body_period_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut body_period_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdltakuri(
        &self,
        body: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdltakuri_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdltakuri_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut period_body_i = 10i32;
            let mut period_upper_i = 10i32;
            let mut period_lower_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut period_body_i as *mut _ as *mut c_void,
                &mut period_upper_i as *mut _ as *mut c_void,
                &mut period_lower_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdltristar(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        body_gap_up: &DeviceBuffer<u8>,
        body_gap_down: &DeviceBuffer<u8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdltristar_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdltristar_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut gap_up_ptr = body_gap_up.as_device_ptr().as_raw();
            let mut gap_down_ptr = body_gap_down.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut period_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut gap_up_ptr as *mut _ as *mut c_void,
                &mut gap_down_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdl2crows(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdl2crows_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdl2crows_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut long_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut long_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdl3blackcrows(
        &self,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdl3blackcrows_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdl3blackcrows_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut shadow_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut shadow_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdl3inside(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdl3inside_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdl3inside_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut long_i = 10i32;
            let mut short_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut long_i as *mut _ as *mut c_void,
                &mut short_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdl3linestrike(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdl3linestrike_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdl3linestrike_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut near_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut near_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdl3outside(
        &self,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdl3outside_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdl3outside_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdl3starsinsouth(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdl3starsinsouth_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdl3starsinsouth_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut body_long_i = 10i32;
            let mut shadow_long_i = 10i32;
            let mut shadow_short_i = 10i32;
            let mut body_short_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut body_long_i as *mut _ as *mut c_void,
                &mut shadow_long_i as *mut _ as *mut c_void,
                &mut shadow_short_i as *mut _ as *mut c_void,
                &mut body_short_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdl3whitesoldiers(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdl3whitesoldiers_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdl3whitesoldiers_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut shadow_i = 10i32;
            let mut near_i = 10i32;
            let mut far_i = 10i32;
            let mut body_short_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut shadow_i as *mut _ as *mut c_void,
                &mut near_i as *mut _ as *mut c_void,
                &mut far_i as *mut _ as *mut c_void,
                &mut body_short_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlabandonedbaby(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlabandonedbaby_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlabandonedbaby_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut body_long_i = 10i32;
            let mut body_doji_i = 10i32;
            let mut body_short_i = 10i32;
            let mut penetration = 0.5f32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut body_long_i as *mut _ as *mut c_void,
                &mut body_doji_i as *mut _ as *mut c_void,
                &mut body_short_i as *mut _ as *mut c_void,
                &mut penetration as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdladvanceblock(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdladvanceblock_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdladvanceblock_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut shadow_short_i = 10i32;
            let mut shadow_long_i = 10i32;
            let mut near_i = 5i32;
            let mut far_i = 5i32;
            let mut body_long_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut shadow_short_i as *mut _ as *mut c_void,
                &mut shadow_long_i as *mut _ as *mut c_void,
                &mut near_i as *mut _ as *mut c_void,
                &mut far_i as *mut _ as *mut c_void,
                &mut body_long_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlbreakaway(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlbreakaway_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlbreakaway_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut body_long_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut body_long_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlconcealbabyswall(
        &self,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlconcealbabyswall_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlconcealbabyswall_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut shadow_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut shadow_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlmathold(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlmathold_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlmathold_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut body_short_i = 10i32;
            let mut body_long_i = 10i32;
            let mut penetration = 0.5f32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut body_short_i as *mut _ as *mut c_void,
                &mut body_long_i as *mut _ as *mut c_void,
                &mut penetration as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlrisefall3methods(
        &self,
        body: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self
            .module
            .get_function("pattern_row_cdlrisefall3methods_u8_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_row_cdlrisefall3methods_u8_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut body_short_i = 10i32;
            let mut body_long_i = 10i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut body_short_i as *mut _ as *mut c_void,
                &mut body_long_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    pub fn pack_matrix_u8_device_into(
        &self,
        d_matrix: &DeviceBuffer<u8>,
        rows: usize,
        cols: usize,
        d_words: &mut DeviceBuffer<u64>,
    ) -> Result<(), CudaPatternRecognitionError> {
        if rows == 0 || cols == 0 {
            return Err(CudaPatternRecognitionError::InvalidInput(
                "rows and cols must be > 0".to_string(),
            ));
        }

        let matrix_len = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaPatternRecognitionError::InvalidInput("rows*cols overflow".to_string()))?;
        if d_matrix.len() < matrix_len {
            return Err(CudaPatternRecognitionError::InvalidInput(
                "matrix buffer too small".to_string(),
            ));
        }

        let words_per_row = cols.div_ceil(64);
        let total_words = rows
            .checked_mul(words_per_row)
            .ok_or_else(|| CudaPatternRecognitionError::InvalidInput("rows*words overflow".to_string()))?;
        if d_words.len() < total_words {
            return Err(CudaPatternRecognitionError::InvalidInput(
                "words buffer too small".to_string(),
            ));
        }

        let func = self
            .module
            .get_function("pattern_pack_u8_to_u64_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_pack_u8_to_u64_kernel",
            })?;

        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(total_words, block_x);

        unsafe {
            let mut matrix_ptr = d_matrix.as_device_ptr().as_raw();
            let mut rows_i = rows as i32;
            let mut cols_i = cols as i32;
            let mut words_per_row_i = words_per_row as i32;
            let mut words_ptr = d_words.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut words_per_row_i as *mut _ as *mut c_void,
                &mut words_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    pub fn pack_matrix_u8_host(
        &self,
        matrix: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<u64>, CudaPatternRecognitionError> {
        if rows == 0 || cols == 0 {
            return Err(CudaPatternRecognitionError::InvalidInput(
                "rows and cols must be > 0".to_string(),
            ));
        }

        let matrix_len = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaPatternRecognitionError::InvalidInput("rows*cols overflow".to_string()))?;
        if matrix.len() != matrix_len {
            return Err(CudaPatternRecognitionError::InvalidInput(format!(
                "matrix length mismatch: expected {}, got {}",
                matrix_len,
                matrix.len()
            )));
        }

        let words_per_row = cols.div_ceil(64);
        let total_words = rows
            .checked_mul(words_per_row)
            .ok_or_else(|| CudaPatternRecognitionError::InvalidInput("rows*words overflow".to_string()))?;

        let d_matrix = DeviceBuffer::from_slice(matrix)?;
        let mut d_words = unsafe { DeviceBuffer::<u64>::uninitialized(total_words) }?;

        self.pack_matrix_u8_device_into(&d_matrix, rows, cols, &mut d_words)?;
        self.synchronize()?;

        let mut host = vec![0u64; total_words];
        d_words.copy_to(&mut host)?;
        Ok(host)
    }

    pub fn matrix_u8_to_f32_device(
        &self,
        d_matrix_u8: &DeviceBuffer<u8>,
        rows: usize,
        cols: usize,
    ) -> Result<DeviceArrayF32, CudaPatternRecognitionError> {
        if rows == 0 || cols == 0 {
            return Err(CudaPatternRecognitionError::InvalidInput(
                "rows and cols must be > 0".to_string(),
            ));
        }

        let len = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaPatternRecognitionError::InvalidInput("rows*cols overflow".to_string()))?;
        if d_matrix_u8.len() < len {
            return Err(CudaPatternRecognitionError::InvalidInput(
                "input matrix buffer too small".to_string(),
            ));
        }

        let mut out = unsafe { DeviceBuffer::<f32>::uninitialized(len) }?;
        let func = self
            .module
            .get_function("pattern_u8_to_f32_kernel")
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol {
                name: "pattern_u8_to_f32_kernel",
            })?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut in_ptr = d_matrix_u8.as_device_ptr().as_raw();
            let mut out_ptr = out.as_device_ptr().as_raw();
            let mut total_i = len as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut in_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
                &mut total_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(DeviceArrayF32 { buf: out, rows, cols })
    }

    pub fn matrix_f32_to_device(
        &self,
        matrix: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<DeviceArrayF32, CudaPatternRecognitionError> {
        if rows == 0 || cols == 0 {
            return Err(CudaPatternRecognitionError::InvalidInput(
                "rows and cols must be > 0".to_string(),
            ));
        }

        let len = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaPatternRecognitionError::InvalidInput("rows*cols overflow".to_string()))?;

        if matrix.len() != len {
            return Err(CudaPatternRecognitionError::InvalidInput(format!(
                "matrix length mismatch: expected {}, got {}",
                len,
                matrix.len()
            )));
        }

        let buf = DeviceBuffer::from_slice(matrix)?;
        Ok(DeviceArrayF32 { buf, rows, cols })
    }
}

fn validate_ohlc_len(
    open: &[f32],
    high: &[f32],
    low: &[f32],
    close: &[f32],
) -> Result<usize, CudaPatternRecognitionError> {
    if open.is_empty() {
        return Err(CudaPatternRecognitionError::InvalidInput(
            "open/high/low/close must be non-empty".to_string(),
        ));
    }

    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(CudaPatternRecognitionError::InvalidInput(format!(
            "length mismatch open={} high={} low={} close={}",
            open.len(),
            high.len(),
            low.len(),
            close.len()
        )));
    }

    Ok(open.len())
}

fn grid_1d_for(n: usize, block_x: u32) -> (GridSize, BlockSize) {
    let gx = ((n as u32).saturating_add(block_x - 1) / block_x).max(1);
    ((gx, 1, 1).into(), (block_x, 1, 1).into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::pattern_recognition::{
        extract_pattern_series, list_patterns, pattern_recognition_with_kernel, PatternRecognitionInput,
    };
    use crate::utilities::enums::Kernel;

    fn sample_ohlc(len: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut open = Vec::with_capacity(len);
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);

        let mut prev_close: f32 = 100.0;
        for i in 0..len {
            let x = i as f32 * 0.013;
            let o = prev_close + x.sin() * 0.7;
            let c = o + (x * 1.3).cos() * 0.4;
            let h = o.max(c) + 0.6 + (x * 0.7).sin().abs() * 0.2;
            let l = o.min(c) - 0.6 - (x * 0.5).cos().abs() * 0.2;
            open.push(o);
            high.push(h);
            low.push(l);
            close.push(c);
            prev_close = c;
        }

        (open, high, low, close)
    }

    fn pattern_row(pattern_id: &str) -> usize {
        list_patterns()
            .iter()
            .find(|spec| spec.id == pattern_id)
            .map(|spec| spec.row_index)
            .unwrap()
    }

    #[test]
    fn feature_kernel_matches_cpu_formula_when_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let (open, high, low, close) = sample_ohlc(512);
        let cuda = CudaPatternRecognition::new(0).unwrap();
        let dev = cuda
            .compute_features_device(&open, &high, &low, &close)
            .unwrap();

        let mut body = vec![0f32; open.len()];
        let mut body_low = vec![0f32; open.len()];
        let mut body_high = vec![0f32; open.len()];
        let mut range = vec![0f32; open.len()];
        let mut upper = vec![0f32; open.len()];
        let mut lower = vec![0f32; open.len()];
        let mut direction = vec![0i8; open.len()];
        let mut gap_up = vec![0u8; open.len()];
        let mut gap_down = vec![0u8; open.len()];

        dev.body.copy_to(&mut body).unwrap();
        dev.body_low.copy_to(&mut body_low).unwrap();
        dev.body_high.copy_to(&mut body_high).unwrap();
        dev.range.copy_to(&mut range).unwrap();
        dev.upper_shadow.copy_to(&mut upper).unwrap();
        dev.lower_shadow.copy_to(&mut lower).unwrap();
        dev.direction.copy_to(&mut direction).unwrap();
        dev.body_gap_up.copy_to(&mut gap_up).unwrap();
        dev.body_gap_down.copy_to(&mut gap_down).unwrap();

        for i in 0..open.len() {
            let o = open[i];
            let h = high[i];
            let l = low[i];
            let c = close[i];

            let body_cpu = (c - o).abs();
            let body_low_cpu = o.min(c);
            let body_high_cpu = o.max(c);
            let range_cpu = h - l;
            let upper_cpu = if c >= o { h - c } else { h - o };
            let lower_cpu = if c >= o { o - l } else { c - l };
            let dir_cpu = if c >= o { 1 } else { -1 };

            assert!((body[i] - body_cpu).abs() <= 1e-5);
            assert!((body_low[i] - body_low_cpu).abs() <= 1e-5);
            assert!((body_high[i] - body_high_cpu).abs() <= 1e-5);
            assert!((range[i] - range_cpu).abs() <= 1e-5);
            assert!((upper[i] - upper_cpu).abs() <= 1e-5);
            assert!((lower[i] - lower_cpu).abs() <= 1e-5);
            assert_eq!(direction[i], dir_cpu);

            if i == 0 {
                assert_eq!(gap_up[i], 0);
                assert_eq!(gap_down[i], 0);
            } else {
                let cur_min = o.min(c);
                let cur_max = o.max(c);
                let prev_min = open[i - 1].min(close[i - 1]);
                let prev_max = open[i - 1].max(close[i - 1]);
                assert_eq!(gap_up[i], if cur_min > prev_max { 1 } else { 0 });
                assert_eq!(gap_down[i], if cur_max < prev_min { 1 } else { 0 });
            }
        }
    }

    #[test]
    fn doji_predicate_kernel_matches_cpu_formula_when_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let (open, high, low, close) = sample_ohlc(257);
        let cuda = CudaPatternRecognition::new(0).unwrap();
        let dev = cuda
            .compute_features_device(&open, &high, &low, &close)
            .unwrap();

        let mut body = vec![0f32; open.len()];
        let mut range = vec![0f32; open.len()];
        dev.body.copy_to(&mut body).unwrap();
        dev.range.copy_to(&mut range).unwrap();

        let threshold = 0.1f32;
        let got = cuda
            .doji_mask_from_features_host(body.as_slice(), range.as_slice(), threshold)
            .unwrap();

        for i in 0..open.len() {
            let b = body[i];
            let r = range[i];
            let hit = b.is_finite() && r.is_finite() && r > 0.0 && b <= threshold * r;
            assert_eq!(got[i], if hit { 1 } else { 0 });
        }
    }

    #[test]
    fn bitmask_kernel_matches_cpu_pack_when_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let rows = 9usize;
        let cols = 173usize;
        let mut matrix = vec![0u8; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let v = ((r * 17 + c * 13 + (c >> 2)) % 11) < 3;
                matrix[r * cols + c] = if v { 1 } else { 0 };
            }
        }

        let cuda = CudaPatternRecognition::new(0).unwrap();
        let got = cuda.pack_matrix_u8_host(matrix.as_slice(), rows, cols).unwrap();

        let words_per_row = cols.div_ceil(64);
        let mut expected = vec![0u64; rows * words_per_row];
        for row in 0..rows {
            for col in 0..cols {
                let value = matrix[row * cols + col];
                if value != 0 {
                    let word = row * words_per_row + (col / 64);
                    let bit = col % 64;
                    expected[word] |= 1u64 << bit;
                }
            }
        }

        assert_eq!(got, expected);
    }

    #[test]
    fn u8_to_f32_kernel_matches_cpu_when_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let rows = 7usize;
        let cols = 129usize;
        let mut matrix = vec![0u8; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                matrix[r * cols + c] = if ((r * 31 + c * 17) % 7) < 3 { 1 } else { 0 };
            }
        }

        let cuda = CudaPatternRecognition::new(0).unwrap();
        let d_u8 = DeviceBuffer::from_slice(matrix.as_slice()).unwrap();
        let d_f32 = cuda.matrix_u8_to_f32_device(&d_u8, rows, cols).unwrap();
        cuda.synchronize().unwrap();

        let mut got = vec![0.0f32; rows * cols];
        d_f32.buf.copy_to(got.as_mut_slice()).unwrap();
        for i in 0..got.len() {
            let expected = if matrix[i] == 0 { 0.0 } else { 1.0 };
            assert_eq!(got[i], expected);
        }
    }

    #[test]
    fn native_supported_rows_match_cpu_matrix_rows_when_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let len = 384usize;
        let (open, high, low, close) = sample_ohlc(len);
        let cuda = CudaPatternRecognition::new(0).unwrap();
        let features = cuda
            .compute_features_device(&open, &high, &low, &close)
            .unwrap();

        let cpu_open: Vec<f64> = open.iter().map(|&v| v as f64).collect();
        let cpu_high: Vec<f64> = high.iter().map(|&v| v as f64).collect();
        let cpu_low: Vec<f64> = low.iter().map(|&v| v as f64).collect();
        let cpu_close: Vec<f64> = close.iter().map(|&v| v as f64).collect();
        let cpu = pattern_recognition_with_kernel(
            &PatternRecognitionInput::with_default_slices(
                cpu_open.as_slice(),
                cpu_high.as_slice(),
                cpu_low.as_slice(),
                cpu_close.as_slice(),
            ),
            Kernel::Auto,
        )
        .unwrap();

        let rows = cpu.rows;
        let cols = cpu.cols;
        let row_map: Vec<(&str, usize)> = CudaPatternRecognition::native_supported_pattern_ids()
            .iter()
            .map(|id| (*id, pattern_row(id)))
            .collect();
        let matrix = cuda
            .compute_native_matrix_host(&features, rows, cols, row_map.as_slice())
            .unwrap();

        let mut mismatches = 0usize;
        let mut total = 0usize;
        for (id, row) in row_map {
            let cpu_row = extract_pattern_series(&cpu, id).unwrap();
            for i in 0..cols {
                total += 1;
                let got = matrix[row * cols + i];
                if got != cpu_row[i] {
                    mismatches += 1;
                }
            }
        }
        let mismatch_ratio = mismatches as f64 / total as f64;
        assert!(
            mismatch_ratio <= 0.01,
            "native CUDA mismatch ratio too high: mismatches={} total={} ratio={:.6}",
            mismatches,
            total,
            mismatch_ratio
        );
    }
}
