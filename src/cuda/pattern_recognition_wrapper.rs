#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{CopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys::{self as cuda, CUfunction};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;
use std::sync::{Arc, Mutex};
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

pub struct DevicePatternBitmaskU64 {
    pub buf: DeviceBuffer<u64>,
    pub rows: usize,
    pub cols: usize,
    pub words_per_row: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}

impl DevicePatternBitmaskU64 {
    #[inline]
    pub fn device_ptr(&self) -> u64 {
        self.buf.as_device_ptr().as_raw() as u64
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.buf.len()
    }
}

pub struct DevicePatternRollingStats {
    pub body_avg10: DeviceBuffer<f32>,
    pub body_avg5: DeviceBuffer<f32>,
    pub upper_avg10: DeviceBuffer<f32>,
    pub lower_avg10: DeviceBuffer<f32>,
    pub max_shadow_avg10: DeviceBuffer<f32>,
    pub belt_shadow_avg10: DeviceBuffer<f32>,
    pub closing_shadow_avg10: DeviceBuffer<f32>,
}

impl DevicePatternRollingStats {
    pub fn len(&self) -> usize {
        self.body_avg10.len()
    }
}

fn row_index_or_neg1(row_map: &[(&str, usize)], pattern_id: &str) -> i32 {
    row_map
        .iter()
        .find(|(id, _)| *id == pattern_id)
        .map(|(_, row)| *row as i32)
        .unwrap_or(-1)
}

fn is_canonical_full_native_row_map(row_map: &[(&str, usize)]) -> bool {
    row_map.len() == NATIVE_SUPPORTED_PATTERN_IDS.len()
        && row_map.iter().enumerate().all(|(idx, (pattern_id, row))| {
            *pattern_id == NATIVE_SUPPORTED_PATTERN_IDS[idx] && *row == idx
        })
}

fn is_simple10_fused_pattern(pattern_id: &str) -> bool {
    matches!(
        pattern_id,
        "cdldoji"
            | "cdldragonflydoji"
            | "cdlgravestonedoji"
            | "cdllongleggeddoji"
            | "cdlmarubozu"
            | "cdlhighwave"
            | "cdllongline"
            | "cdlshortline"
            | "cdlspinningtop"
    )
}

fn is_two_bar_body10_fused_pattern(pattern_id: &str) -> bool {
    matches!(
        pattern_id,
        "cdldojistar" | "cdlharami" | "cdlharamicross" | "cdlhomingpigeon"
    )
}

fn is_single_bar_shadow_fused_pattern(pattern_id: &str) -> bool {
    matches!(
        pattern_id,
        "cdlhammer"
            | "cdlhangingman"
            | "cdlinvertedhammer"
            | "cdlshootingstar"
            | "cdltakuri"
            | "cdlrickshawman"
    )
}

fn is_directional_shadow_fused_pattern(pattern_id: &str) -> bool {
    matches!(
        pattern_id,
        "cdlbelthold" | "cdlclosingmarubozu" | "cdlkicking" | "cdlkickingbylength"
    )
}

fn is_star3_fused_pattern(pattern_id: &str) -> bool {
    matches!(
        pattern_id,
        "cdleveningdojistar" | "cdleveningstar" | "cdlmorningdojistar" | "cdlmorningstar"
    )
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
    kernel_handles: Mutex<HashMap<&'static str, CUfunction>>,
    device_id: u32,
    #[cfg(test)]
    _test_lock: crate::cuda::CudaTestLock,
}

impl CudaPatternRecognition {
    pub fn new(device_id: usize) -> Result<Self, CudaPatternRecognitionError> {
        #[cfg(test)]
        let test_lock = crate::cuda::cuda_test_lock();

        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/pattern_recognition_kernel.ptx"));
        let module = crate::load_cuda_embedded_module!("pattern_recognition_kernel")?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            context,
            kernel_handles: Mutex::new(HashMap::new()),
            device_id: device_id as u32,
            #[cfg(test)]
            _test_lock: test_lock,
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

    fn allocate_rolling_stat_buffers(
        &self,
        len: usize,
    ) -> Result<DevicePatternRollingStats, CudaPatternRecognitionError> {
        if len == 0 {
            return Err(CudaPatternRecognitionError::InvalidInput(
                "len must be > 0".to_string(),
            ));
        }

        Ok(DevicePatternRollingStats {
            body_avg10: unsafe { DeviceBuffer::<f32>::uninitialized(len) }?,
            body_avg5: unsafe { DeviceBuffer::<f32>::uninitialized(len) }?,
            upper_avg10: unsafe { DeviceBuffer::<f32>::uninitialized(len) }?,
            lower_avg10: unsafe { DeviceBuffer::<f32>::uninitialized(len) }?,
            max_shadow_avg10: unsafe { DeviceBuffer::<f32>::uninitialized(len) }?,
            belt_shadow_avg10: unsafe { DeviceBuffer::<f32>::uninitialized(len) }?,
            closing_shadow_avg10: unsafe { DeviceBuffer::<f32>::uninitialized(len) }?,
        })
    }

    pub fn compute_features_device(
        &self,
        open: &[f32],
        high: &[f32],
        low: &[f32],
        close: &[f32],
    ) -> Result<DevicePatternFeatures, CudaPatternRecognitionError> {
        let out = self.compute_features_device_async(open, high, low, close)?;
        self.synchronize()?;

        Ok(out)
    }

    pub fn compute_features_device_async(
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

        Ok(out)
    }

    fn cached_function(
        &self,
        name: &'static str,
    ) -> Result<CUfunction, CudaPatternRecognitionError> {
        let mut cache = self.kernel_handles.lock().map_err(|_| {
            CudaPatternRecognitionError::InvalidInput("kernel handle cache poisoned".to_string())
        })?;
        if let Some(&func) = cache.get(name) {
            return Ok(func);
        }

        let func = self
            .module
            .get_function(name)
            .map_err(|_| CudaPatternRecognitionError::MissingKernelSymbol { name })?
            .to_raw();
        cache.insert(name, func);
        Ok(func)
    }

    unsafe fn launch_raw_function<G, B>(
        &self,
        func: CUfunction,
        grid_size: G,
        block_size: B,
        shared_mem_bytes: u32,
        args: &[*mut c_void],
    ) -> Result<(), CudaPatternRecognitionError>
    where
        G: Into<GridSize>,
        B: Into<BlockSize>,
    {
        let grid_size: GridSize = grid_size.into();
        let block_size: BlockSize = block_size.into();

        let result = cuda::cuLaunchKernel(
            func,
            grid_size.x,
            grid_size.y,
            grid_size.z,
            block_size.x,
            block_size.y,
            block_size.z,
            shared_mem_bytes,
            self.stream.as_inner(),
            args.as_ptr() as *mut _,
            ptr::null_mut(),
        );
        if result == cuda::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(unsafe { std::mem::transmute::<u32, cust::error::CudaError>(result as u32) }.into())
        }
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

        let func = self.cached_function("pattern_features_kernel_f32")?;

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
            self.launch_raw_function(func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn rolling_mean_f32_device_into(
        &self,
        input: &DeviceBuffer<f32>,
        len: usize,
        period: i32,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaPatternRecognitionError> {
        if len == 0 || period <= 0 {
            return Err(CudaPatternRecognitionError::InvalidInput(
                "len must be > 0 and period must be > 0".to_string(),
            ));
        }
        if input.len() < len || out.len() < len {
            return Err(CudaPatternRecognitionError::InvalidInput(
                "buffer too small for rolling mean".to_string(),
            ));
        }

        let func = self.cached_function("pattern_rolling_mean_f32_kernel")?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut in_ptr = input.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut period_i = period;
            let mut out_ptr = out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut in_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.launch_raw_function(func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn rolling_max_shadow_mean_f32_device_into(
        &self,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        len: usize,
        period: i32,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaPatternRecognitionError> {
        if len == 0 || period <= 0 {
            return Err(CudaPatternRecognitionError::InvalidInput(
                "len must be > 0 and period must be > 0".to_string(),
            ));
        }
        if upper.len() < len || lower.len() < len || out.len() < len {
            return Err(CudaPatternRecognitionError::InvalidInput(
                "buffer too small for rolling max shadow mean".to_string(),
            ));
        }

        let func = self.cached_function("pattern_rolling_max_shadow_mean_f32_kernel")?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut period_i = period;
            let mut out_ptr = out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.launch_raw_function(func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn rolling_stats_10_device_into(
        &self,
        features: &DevicePatternFeatures,
        len: usize,
        out: &mut DevicePatternRollingStats,
    ) -> Result<(), CudaPatternRecognitionError> {
        if features.len() < len || out.len() < len {
            return Err(CudaPatternRecognitionError::InvalidInput(
                "buffer too small for rolling stats".to_string(),
            ));
        }

        let func = self.cached_function("pattern_rolling_stats_10_f32_kernel")?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = features.body.as_device_ptr().as_raw();
            let mut upper_ptr = features.upper_shadow.as_device_ptr().as_raw();
            let mut lower_ptr = features.lower_shadow.as_device_ptr().as_raw();
            let mut direction_ptr = features.direction.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut body_avg_ptr = out.body_avg10.as_device_ptr().as_raw();
            let mut body_avg5_ptr = out.body_avg5.as_device_ptr().as_raw();
            let mut upper_avg_ptr = out.upper_avg10.as_device_ptr().as_raw();
            let mut lower_avg_ptr = out.lower_avg10.as_device_ptr().as_raw();
            let mut max_shadow_avg_ptr = out.max_shadow_avg10.as_device_ptr().as_raw();
            let mut belt_shadow_avg_ptr = out.belt_shadow_avg10.as_device_ptr().as_raw();
            let mut closing_shadow_avg_ptr = out.closing_shadow_avg10.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut direction_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut body_avg_ptr as *mut _ as *mut c_void,
                &mut body_avg5_ptr as *mut _ as *mut c_void,
                &mut upper_avg_ptr as *mut _ as *mut c_void,
                &mut lower_avg_ptr as *mut _ as *mut c_void,
                &mut max_shadow_avg_ptr as *mut _ as *mut c_void,
                &mut belt_shadow_avg_ptr as *mut _ as *mut c_void,
                &mut closing_shadow_avg_ptr as *mut _ as *mut c_void,
            ];
            self.launch_raw_function(func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn compute_shared_rolling_stats_device(
        &self,
        features: &DevicePatternFeatures,
        len: usize,
    ) -> Result<DevicePatternRollingStats, CudaPatternRecognitionError> {
        if features.len() < len {
            return Err(CudaPatternRecognitionError::InvalidInput(
                "features length mismatch for rolling stats".to_string(),
            ));
        }

        let mut out = self.allocate_rolling_stat_buffers(len)?;
        self.rolling_stats_10_device_into(features, len, &mut out)?;
        Ok(out)
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

        let func = self.cached_function("pattern_doji_predicate_kernel_f32")?;

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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let mut d_matrix = if is_canonical_full_native_row_map(row_map) {
            unsafe { DeviceBuffer::<u8>::uninitialized(total) }?
        } else {
            DeviceBuffer::<u8>::zeroed(total)?
        };
        let rolling = self.compute_shared_rolling_stats_device(features, cols)?;
        self.launch_simple10_rows(features, &rolling, cols, &mut d_matrix, cols, row_map)?;
        self.launch_two_bar_body10_rows(features, &rolling, cols, &mut d_matrix, cols, row_map)?;
        self.launch_single_bar_shadow_rows(features, &rolling, cols, &mut d_matrix, cols, row_map)?;
        self.launch_directional_shadow_rows(
            features,
            &rolling,
            cols,
            &mut d_matrix,
            cols,
            row_map,
        )?;
        self.launch_star3_rows(features, &rolling, cols, &mut d_matrix, cols, row_map)?;

        for (pattern_id, row) in row_map {
            if is_simple10_fused_pattern(pattern_id) {
                continue;
            }
            if is_two_bar_body10_fused_pattern(pattern_id) {
                continue;
            }
            if is_single_bar_shadow_fused_pattern(pattern_id) {
                continue;
            }
            if is_directional_shadow_fused_pattern(pattern_id) {
                continue;
            }
            if is_star3_fused_pattern(pattern_id) {
                continue;
            }
            self.launch_pattern_row(
                features,
                &rolling,
                cols,
                &mut d_matrix,
                cols,
                *row,
                pattern_id,
            )?;
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

    pub fn compute_native_matrix_device_from_device_inputs(
        &self,
        d_open: &DeviceBuffer<f32>,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        len: usize,
    ) -> Result<DeviceBuffer<u8>, CudaPatternRecognitionError> {
        if len == 0 {
            return Err(CudaPatternRecognitionError::InvalidInput(
                "len must be > 0".to_string(),
            ));
        }
        if d_open.len() < len || d_high.len() < len || d_low.len() < len || d_close.len() < len {
            return Err(CudaPatternRecognitionError::InvalidInput(
                "device input buffer too small for len".to_string(),
            ));
        }

        let mut features = self.allocate_feature_buffers(len)?;
        self.compute_features_device_into(d_open, d_high, d_low, d_close, len, &mut features)?;

        let native_ids = Self::native_supported_pattern_ids();
        let rows = native_ids.len();
        let row_map: Vec<(&str, usize)> = native_ids
            .iter()
            .enumerate()
            .map(|(row, id)| (*id, row))
            .collect();
        self.compute_native_matrix_device(&features, rows, len, row_map.as_slice())
    }

    pub fn compute_native_matrix_device_from_host_inputs(
        &self,
        open: &[f32],
        high: &[f32],
        low: &[f32],
        close: &[f32],
    ) -> Result<DeviceBuffer<u8>, CudaPatternRecognitionError> {
        let len = validate_ohlc_len(open, high, low, close)?;
        let d_open = DeviceBuffer::from_slice(open)?;
        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        self.compute_native_matrix_device_from_device_inputs(
            &d_open, &d_high, &d_low, &d_close, len,
        )
    }

    pub fn compute_native_matrix_f32_device_from_device_inputs(
        &self,
        d_open: &DeviceBuffer<f32>,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        len: usize,
    ) -> Result<DeviceArrayF32, CudaPatternRecognitionError> {
        let native_ids = Self::native_supported_pattern_ids();
        let rows = native_ids.len();
        let d_u8 = self
            .compute_native_matrix_device_from_device_inputs(d_open, d_high, d_low, d_close, len)?;
        self.matrix_u8_to_f32_device(&d_u8, rows, len)
    }

    pub fn compute_native_matrix_f32_device_from_host_inputs(
        &self,
        open: &[f32],
        high: &[f32],
        low: &[f32],
        close: &[f32],
    ) -> Result<DeviceArrayF32, CudaPatternRecognitionError> {
        let len = validate_ohlc_len(open, high, low, close)?;
        let d_u8 = self.compute_native_matrix_device_from_host_inputs(open, high, low, close)?;
        let rows = Self::native_supported_pattern_ids().len();
        self.matrix_u8_to_f32_device(&d_u8, rows, len)
    }

    pub fn compute_native_matrix_bitmask_u64_device_from_device_inputs(
        &self,
        d_open: &DeviceBuffer<f32>,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        len: usize,
    ) -> Result<DevicePatternBitmaskU64, CudaPatternRecognitionError> {
        let native_ids = Self::native_supported_pattern_ids();
        let rows = native_ids.len();
        let d_u8 = self
            .compute_native_matrix_device_from_device_inputs(d_open, d_high, d_low, d_close, len)?;
        let words_per_row = len.div_ceil(64);
        let total_words = rows.checked_mul(words_per_row).ok_or_else(|| {
            CudaPatternRecognitionError::InvalidInput("rows*words overflow".to_string())
        })?;
        let mut d_words = unsafe { DeviceBuffer::<u64>::uninitialized(total_words) }?;
        self.pack_matrix_u8_device_into(&d_u8, rows, len, &mut d_words)?;
        Ok(DevicePatternBitmaskU64 {
            buf: d_words,
            rows,
            cols: len,
            words_per_row,
            ctx: self.context.clone(),
            device_id: self.device_id,
        })
    }

    pub fn compute_native_matrix_bitmask_u64_device_from_host_inputs(
        &self,
        open: &[f32],
        high: &[f32],
        low: &[f32],
        close: &[f32],
    ) -> Result<DevicePatternBitmaskU64, CudaPatternRecognitionError> {
        let len = validate_ohlc_len(open, high, low, close)?;
        let d_open = DeviceBuffer::from_slice(open)?;
        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        self.compute_native_matrix_bitmask_u64_device_from_device_inputs(
            &d_open, &d_high, &d_low, &d_close, len,
        )
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

    fn launch_simple10_rows(
        &self,
        features: &DevicePatternFeatures,
        rolling: &DevicePatternRollingStats,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row_map: &[(&str, usize)],
    ) -> Result<(), CudaPatternRecognitionError> {
        let row_cdldoji = row_index_or_neg1(row_map, "cdldoji");
        let row_cdldragonflydoji = row_index_or_neg1(row_map, "cdldragonflydoji");
        let row_cdlgravestonedoji = row_index_or_neg1(row_map, "cdlgravestonedoji");
        let row_cdllongleggeddoji = row_index_or_neg1(row_map, "cdllongleggeddoji");
        let row_cdlmarubozu = row_index_or_neg1(row_map, "cdlmarubozu");
        let row_cdlhighwave = row_index_or_neg1(row_map, "cdlhighwave");
        let row_cdllongline = row_index_or_neg1(row_map, "cdllongline");
        let row_cdlshortline = row_index_or_neg1(row_map, "cdlshortline");
        let row_cdlspinningtop = row_index_or_neg1(row_map, "cdlspinningtop");

        if row_cdldoji < 0
            && row_cdldragonflydoji < 0
            && row_cdlgravestonedoji < 0
            && row_cdllongleggeddoji < 0
            && row_cdlmarubozu < 0
            && row_cdlhighwave < 0
            && row_cdllongline < 0
            && row_cdlshortline < 0
            && row_cdlspinningtop < 0
        {
            return Ok(());
        }

        let func = self.cached_function("pattern_rows_simple10_u8_kernel")?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = features.body.as_device_ptr().as_raw();
            let mut body_avg_ptr = rolling.body_avg10.as_device_ptr().as_raw();
            let mut upper_ptr = features.upper_shadow.as_device_ptr().as_raw();
            let mut lower_ptr = features.lower_shadow.as_device_ptr().as_raw();
            let mut upper_avg_ptr = rolling.upper_avg10.as_device_ptr().as_raw();
            let mut max_shadow_avg_ptr = rolling.max_shadow_avg10.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_cdldoji_i = row_cdldoji;
            let mut row_cdldragonflydoji_i = row_cdldragonflydoji;
            let mut row_cdlgravestonedoji_i = row_cdlgravestonedoji;
            let mut row_cdllongleggeddoji_i = row_cdllongleggeddoji;
            let mut row_cdlmarubozu_i = row_cdlmarubozu;
            let mut row_cdlhighwave_i = row_cdlhighwave;
            let mut row_cdllongline_i = row_cdllongline;
            let mut row_cdlshortline_i = row_cdlshortline;
            let mut row_cdlspinningtop_i = row_cdlspinningtop;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_avg_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut upper_avg_ptr as *mut _ as *mut c_void,
                &mut max_shadow_avg_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_cdldoji_i as *mut _ as *mut c_void,
                &mut row_cdldragonflydoji_i as *mut _ as *mut c_void,
                &mut row_cdlgravestonedoji_i as *mut _ as *mut c_void,
                &mut row_cdllongleggeddoji_i as *mut _ as *mut c_void,
                &mut row_cdlmarubozu_i as *mut _ as *mut c_void,
                &mut row_cdlhighwave_i as *mut _ as *mut c_void,
                &mut row_cdllongline_i as *mut _ as *mut c_void,
                &mut row_cdlshortline_i as *mut _ as *mut c_void,
                &mut row_cdlspinningtop_i as *mut _ as *mut c_void,
            ];
            self.launch_raw_function(func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_two_bar_body10_rows(
        &self,
        features: &DevicePatternFeatures,
        rolling: &DevicePatternRollingStats,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row_map: &[(&str, usize)],
    ) -> Result<(), CudaPatternRecognitionError> {
        let row_cdldojistar = row_index_or_neg1(row_map, "cdldojistar");
        let row_cdlharami = row_index_or_neg1(row_map, "cdlharami");
        let row_cdlharamicross = row_index_or_neg1(row_map, "cdlharamicross");
        let row_cdlhomingpigeon = row_index_or_neg1(row_map, "cdlhomingpigeon");

        if row_cdldojistar < 0
            && row_cdlharami < 0
            && row_cdlharamicross < 0
            && row_cdlhomingpigeon < 0
        {
            return Ok(());
        }

        let func = self.cached_function("pattern_rows_two_bar_body10_u8_kernel")?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = features.body.as_device_ptr().as_raw();
            let mut body_avg_ptr = rolling.body_avg10.as_device_ptr().as_raw();
            let mut body_low_ptr = features.body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = features.body_high.as_device_ptr().as_raw();
            let mut direction_ptr = features.direction.as_device_ptr().as_raw();
            let mut gap_up_ptr = features.body_gap_up.as_device_ptr().as_raw();
            let mut gap_down_ptr = features.body_gap_down.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_cdldojistar_i = row_cdldojistar;
            let mut row_cdlharami_i = row_cdlharami;
            let mut row_cdlharamicross_i = row_cdlharamicross;
            let mut row_cdlhomingpigeon_i = row_cdlhomingpigeon;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_avg_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut direction_ptr as *mut _ as *mut c_void,
                &mut gap_up_ptr as *mut _ as *mut c_void,
                &mut gap_down_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_cdldojistar_i as *mut _ as *mut c_void,
                &mut row_cdlharami_i as *mut _ as *mut c_void,
                &mut row_cdlharamicross_i as *mut _ as *mut c_void,
                &mut row_cdlhomingpigeon_i as *mut _ as *mut c_void,
            ];
            self.launch_raw_function(func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_single_bar_shadow_rows(
        &self,
        features: &DevicePatternFeatures,
        rolling: &DevicePatternRollingStats,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row_map: &[(&str, usize)],
    ) -> Result<(), CudaPatternRecognitionError> {
        let row_cdlhammer = row_index_or_neg1(row_map, "cdlhammer");
        let row_cdlhangingman = row_index_or_neg1(row_map, "cdlhangingman");
        let row_cdlinvertedhammer = row_index_or_neg1(row_map, "cdlinvertedhammer");
        let row_cdlshootingstar = row_index_or_neg1(row_map, "cdlshootingstar");
        let row_cdltakuri = row_index_or_neg1(row_map, "cdltakuri");
        let row_cdlrickshawman = row_index_or_neg1(row_map, "cdlrickshawman");

        if row_cdlhammer < 0
            && row_cdlhangingman < 0
            && row_cdlinvertedhammer < 0
            && row_cdlshootingstar < 0
            && row_cdltakuri < 0
            && row_cdlrickshawman < 0
        {
            return Ok(());
        }

        let func = self.cached_function("pattern_rows_single_bar_shadow_u8_kernel")?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = features.body.as_device_ptr().as_raw();
            let mut body_low_ptr = features.body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = features.body_high.as_device_ptr().as_raw();
            let mut body_avg_ptr = rolling.body_avg10.as_device_ptr().as_raw();
            let mut body_avg5_ptr = rolling.body_avg5.as_device_ptr().as_raw();
            let mut upper_ptr = features.upper_shadow.as_device_ptr().as_raw();
            let mut lower_ptr = features.lower_shadow.as_device_ptr().as_raw();
            let mut upper_avg_ptr = rolling.upper_avg10.as_device_ptr().as_raw();
            let mut lower_avg_ptr = rolling.lower_avg10.as_device_ptr().as_raw();
            let mut gap_up_ptr = features.body_gap_up.as_device_ptr().as_raw();
            let mut gap_down_ptr = features.body_gap_down.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_cdlhammer_i = row_cdlhammer;
            let mut row_cdlhangingman_i = row_cdlhangingman;
            let mut row_cdlinvertedhammer_i = row_cdlinvertedhammer;
            let mut row_cdlshootingstar_i = row_cdlshootingstar;
            let mut row_cdltakuri_i = row_cdltakuri;
            let mut row_cdlrickshawman_i = row_cdlrickshawman;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut body_avg_ptr as *mut _ as *mut c_void,
                &mut body_avg5_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut upper_avg_ptr as *mut _ as *mut c_void,
                &mut lower_avg_ptr as *mut _ as *mut c_void,
                &mut gap_up_ptr as *mut _ as *mut c_void,
                &mut gap_down_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_cdlhammer_i as *mut _ as *mut c_void,
                &mut row_cdlhangingman_i as *mut _ as *mut c_void,
                &mut row_cdlinvertedhammer_i as *mut _ as *mut c_void,
                &mut row_cdlshootingstar_i as *mut _ as *mut c_void,
                &mut row_cdltakuri_i as *mut _ as *mut c_void,
                &mut row_cdlrickshawman_i as *mut _ as *mut c_void,
            ];
            self.launch_raw_function(func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_directional_shadow_rows(
        &self,
        features: &DevicePatternFeatures,
        rolling: &DevicePatternRollingStats,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row_map: &[(&str, usize)],
    ) -> Result<(), CudaPatternRecognitionError> {
        let row_cdlbelthold = row_index_or_neg1(row_map, "cdlbelthold");
        let row_cdlclosingmarubozu = row_index_or_neg1(row_map, "cdlclosingmarubozu");
        let row_cdlkicking = row_index_or_neg1(row_map, "cdlkicking");
        let row_cdlkickingbylength = row_index_or_neg1(row_map, "cdlkickingbylength");

        if row_cdlbelthold < 0
            && row_cdlclosingmarubozu < 0
            && row_cdlkicking < 0
            && row_cdlkickingbylength < 0
        {
            return Ok(());
        }

        let func = self.cached_function("pattern_rows_directional_shadow_u8_kernel")?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = features.body.as_device_ptr().as_raw();
            let mut body_avg_ptr = rolling.body_avg10.as_device_ptr().as_raw();
            let mut upper_ptr = features.upper_shadow.as_device_ptr().as_raw();
            let mut lower_ptr = features.lower_shadow.as_device_ptr().as_raw();
            let mut max_shadow_avg_ptr = rolling.max_shadow_avg10.as_device_ptr().as_raw();
            let mut belt_shadow_avg_ptr = rolling.belt_shadow_avg10.as_device_ptr().as_raw();
            let mut closing_shadow_avg_ptr = rolling.closing_shadow_avg10.as_device_ptr().as_raw();
            let mut direction_ptr = features.direction.as_device_ptr().as_raw();
            let mut gap_up_ptr = features.body_gap_up.as_device_ptr().as_raw();
            let mut gap_down_ptr = features.body_gap_down.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_cdlbelthold_i = row_cdlbelthold;
            let mut row_cdlclosingmarubozu_i = row_cdlclosingmarubozu;
            let mut row_cdlkicking_i = row_cdlkicking;
            let mut row_cdlkickingbylength_i = row_cdlkickingbylength;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_avg_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut max_shadow_avg_ptr as *mut _ as *mut c_void,
                &mut belt_shadow_avg_ptr as *mut _ as *mut c_void,
                &mut closing_shadow_avg_ptr as *mut _ as *mut c_void,
                &mut direction_ptr as *mut _ as *mut c_void,
                &mut gap_up_ptr as *mut _ as *mut c_void,
                &mut gap_down_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_cdlbelthold_i as *mut _ as *mut c_void,
                &mut row_cdlclosingmarubozu_i as *mut _ as *mut c_void,
                &mut row_cdlkicking_i as *mut _ as *mut c_void,
                &mut row_cdlkickingbylength_i as *mut _ as *mut c_void,
            ];
            self.launch_raw_function(func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_star3_rows(
        &self,
        features: &DevicePatternFeatures,
        rolling: &DevicePatternRollingStats,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row_map: &[(&str, usize)],
    ) -> Result<(), CudaPatternRecognitionError> {
        let row_cdleveningdojistar = row_index_or_neg1(row_map, "cdleveningdojistar");
        let row_cdleveningstar = row_index_or_neg1(row_map, "cdleveningstar");
        let row_cdlmorningdojistar = row_index_or_neg1(row_map, "cdlmorningdojistar");
        let row_cdlmorningstar = row_index_or_neg1(row_map, "cdlmorningstar");

        if row_cdleveningdojistar < 0
            && row_cdleveningstar < 0
            && row_cdlmorningdojistar < 0
            && row_cdlmorningstar < 0
        {
            return Ok(());
        }

        let func = self.cached_function("pattern_rows_star3_u8_kernel")?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = features.body.as_device_ptr().as_raw();
            let mut body_low_ptr = features.body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = features.body_high.as_device_ptr().as_raw();
            let mut body_avg_ptr = rolling.body_avg10.as_device_ptr().as_raw();
            let mut direction_ptr = features.direction.as_device_ptr().as_raw();
            let mut gap_up_ptr = features.body_gap_up.as_device_ptr().as_raw();
            let mut gap_down_ptr = features.body_gap_down.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut penetration = 0.3f32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_cdleveningdojistar_i = row_cdleveningdojistar;
            let mut row_cdleveningstar_i = row_cdleveningstar;
            let mut row_cdlmorningdojistar_i = row_cdlmorningdojistar;
            let mut row_cdlmorningstar_i = row_cdlmorningstar;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut body_avg_ptr as *mut _ as *mut c_void,
                &mut direction_ptr as *mut _ as *mut c_void,
                &mut gap_up_ptr as *mut _ as *mut c_void,
                &mut gap_down_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut penetration as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_cdleveningdojistar_i as *mut _ as *mut c_void,
                &mut row_cdleveningstar_i as *mut _ as *mut c_void,
                &mut row_cdlmorningdojistar_i as *mut _ as *mut c_void,
                &mut row_cdlmorningstar_i as *mut _ as *mut c_void,
            ];
            self.launch_raw_function(func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_pattern_row(
        &self,
        features: &DevicePatternFeatures,
        rolling: &DevicePatternRollingStats,
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
            "cdldoji" => {
                self.launch_row_cdldoji(&features.body, &rolling.body_avg10, len, matrix, cols, row)
            }
            "cdldojistar" => self.launch_row_cdldojistar(
                &features.body,
                &rolling.body_avg10,
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
                &rolling.body_avg10,
                &features.upper_shadow,
                &features.lower_shadow,
                &rolling.max_shadow_avg10,
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
                &rolling.body_avg10,
                &features.upper_shadow,
                &features.lower_shadow,
                &rolling.upper_avg10,
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
                &rolling.body_avg10,
                &features.body_low,
                &features.body_high,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlharamicross" => self.launch_row_cdlharami(
                &features.body,
                &rolling.body_avg10,
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
                &rolling.body_avg10,
                &features.upper_shadow,
                &features.lower_shadow,
                &rolling.upper_avg10,
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
                &rolling.body_avg10,
                &features.upper_shadow,
                &rolling.upper_avg10,
                &features.lower_shadow,
                &rolling.lower_avg10,
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
                &rolling.body_avg10,
                &features.upper_shadow,
                &features.lower_shadow,
                &rolling.upper_avg10,
                len,
                matrix,
                cols,
                row,
            ),
            "cdllongline" => self.launch_row_cdllongline(
                &features.body,
                &rolling.body_avg10,
                &features.upper_shadow,
                &features.lower_shadow,
                &rolling.upper_avg10,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlmarubozu" => self.launch_row_cdlmarubozu(
                &features.body,
                &rolling.body_avg10,
                &features.upper_shadow,
                &features.lower_shadow,
                &rolling.upper_avg10,
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
                &rolling.body_avg10,
                &features.upper_shadow,
                &features.lower_shadow,
                &rolling.upper_avg10,
                len,
                matrix,
                cols,
                row,
            ),
            "cdlspinningtop" => self.launch_row_cdlspinningtop(
                &features.body,
                &rolling.body_avg10,
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
        body_avg10: &DeviceBuffer<f32>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self.cached_function("pattern_row_cdldoji_u8_kernel")?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_avg_ptr = body_avg10.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_avg_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.launch_raw_function(func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdldragonflydoji(
        &self,
        body: &DeviceBuffer<f32>,
        body_avg10: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        max_shadow_avg10: &DeviceBuffer<f32>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self.cached_function("pattern_row_cdldragonflydoji_u8_kernel")?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_avg_ptr = body_avg10.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut shadow_avg_ptr = max_shadow_avg10.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_avg_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut shadow_avg_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.launch_raw_function(func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlgravestonedoji(
        &self,
        body: &DeviceBuffer<f32>,
        body_avg10: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        upper_avg10: &DeviceBuffer<f32>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self.cached_function("pattern_row_cdlgravestonedoji_u8_kernel")?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_avg_ptr = body_avg10.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut upper_avg_ptr = upper_avg10.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_avg_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut upper_avg_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.launch_raw_function(func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdllongleggeddoji(
        &self,
        body: &DeviceBuffer<f32>,
        body_avg10: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        upper_avg10: &DeviceBuffer<f32>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self.cached_function("pattern_row_cdllongleggeddoji_u8_kernel")?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_avg_ptr = body_avg10.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut upper_avg_ptr = upper_avg10.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_avg_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut upper_avg_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.launch_raw_function(func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlmarubozu(
        &self,
        body: &DeviceBuffer<f32>,
        body_avg10: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        upper_avg10: &DeviceBuffer<f32>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self.cached_function("pattern_row_cdlmarubozu_u8_kernel")?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_avg_ptr = body_avg10.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut upper_avg_ptr = upper_avg10.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_avg_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut upper_avg_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlbelthold_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlclosingmarubozu_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlhammer_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlhangingman_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlrickshawman_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlmatchinglow_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlinneck_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlonneck_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlpiercing_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlthrusting_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdleveningdojistar_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdleveningstar_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlmorningdojistar_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlmorningstar_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlgapsidesidewhite_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlkicking_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlkickingbylength_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlidentical3crows_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlsticksandwich_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlseparatinglines_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlcounterattack_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdldarkcloudcover_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlxsidegap3methods_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlupsidegap2crows_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlunique3river_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdltasukigap_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlladderbottom_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlstalledpattern_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlhikkake_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlhikkakemod_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
        }
        Ok(())
    }

    fn launch_row_cdldojistar(
        &self,
        body: &DeviceBuffer<f32>,
        body_avg10: &DeviceBuffer<f32>,
        direction: &DeviceBuffer<i8>,
        body_gap_up: &DeviceBuffer<u8>,
        body_gap_down: &DeviceBuffer<u8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self.cached_function("pattern_row_cdldojistar_u8_kernel")?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_avg_ptr = body_avg10.as_device_ptr().as_raw();
            let mut dir_ptr = direction.as_device_ptr().as_raw();
            let mut gap_up_ptr = body_gap_up.as_device_ptr().as_raw();
            let mut gap_down_ptr = body_gap_down.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_avg_ptr as *mut _ as *mut c_void,
                &mut dir_ptr as *mut _ as *mut c_void,
                &mut gap_up_ptr as *mut _ as *mut c_void,
                &mut gap_down_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlengulfing_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlharami(
        &self,
        body: &DeviceBuffer<f32>,
        body_avg10: &DeviceBuffer<f32>,
        body_low: &DeviceBuffer<f32>,
        body_high: &DeviceBuffer<f32>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self.cached_function("pattern_row_cdlharami_u8_kernel")?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_avg_ptr = body_avg10.as_device_ptr().as_raw();
            let mut body_low_ptr = body_low.as_device_ptr().as_raw();
            let mut body_high_ptr = body_high.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_avg_ptr as *mut _ as *mut c_void,
                &mut body_low_ptr as *mut _ as *mut c_void,
                &mut body_high_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.launch_raw_function(func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlhighwave(
        &self,
        body: &DeviceBuffer<f32>,
        body_avg10: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        upper_avg10: &DeviceBuffer<f32>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self.cached_function("pattern_row_cdlhighwave_u8_kernel")?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_avg_ptr = body_avg10.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut upper_avg_ptr = upper_avg10.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_avg_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut upper_avg_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlhomingpigeon_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlinvertedhammer(
        &self,
        body: &DeviceBuffer<f32>,
        body_avg10: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        upper_avg10: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        lower_avg10: &DeviceBuffer<f32>,
        body_gap_down: &DeviceBuffer<u8>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self.cached_function("pattern_row_cdlinvertedhammer_u8_kernel")?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_avg_ptr = body_avg10.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut upper_avg_ptr = upper_avg10.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut lower_avg_ptr = lower_avg10.as_device_ptr().as_raw();
            let mut gap_ptr = body_gap_down.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_avg_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut upper_avg_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut lower_avg_ptr as *mut _ as *mut c_void,
                &mut gap_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.launch_raw_function(func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdllongline(
        &self,
        body: &DeviceBuffer<f32>,
        body_avg10: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        upper_avg10: &DeviceBuffer<f32>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self.cached_function("pattern_row_cdllongline_u8_kernel")?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_avg_ptr = body_avg10.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut upper_avg_ptr = upper_avg10.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_avg_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut upper_avg_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlshootingstar_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlshortline(
        &self,
        body: &DeviceBuffer<f32>,
        body_avg10: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        upper_avg10: &DeviceBuffer<f32>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self.cached_function("pattern_row_cdlshortline_u8_kernel")?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_avg_ptr = body_avg10.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut upper_avg_ptr = upper_avg10.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_avg_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut upper_avg_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.launch_raw_function(func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_row_cdlspinningtop(
        &self,
        body: &DeviceBuffer<f32>,
        body_avg10: &DeviceBuffer<f32>,
        upper: &DeviceBuffer<f32>,
        lower: &DeviceBuffer<f32>,
        len: usize,
        matrix: &mut DeviceBuffer<u8>,
        cols: usize,
        row: usize,
    ) -> Result<(), CudaPatternRecognitionError> {
        let func = self.cached_function("pattern_row_cdlspinningtop_u8_kernel")?;
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x);

        unsafe {
            let mut body_ptr = body.as_device_ptr().as_raw();
            let mut body_avg_ptr = body_avg10.as_device_ptr().as_raw();
            let mut upper_ptr = upper.as_device_ptr().as_raw();
            let mut lower_ptr = lower.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut matrix_ptr = matrix.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut row_i = row as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut body_ptr as *mut _ as *mut c_void,
                &mut body_avg_ptr as *mut _ as *mut c_void,
                &mut upper_ptr as *mut _ as *mut c_void,
                &mut lower_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut matrix_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut row_i as *mut _ as *mut c_void,
            ];
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdltakuri_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdltristar_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdl2crows_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdl3blackcrows_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdl3inside_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdl3linestrike_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdl3outside_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdl3starsinsouth_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdl3whitesoldiers_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlabandonedbaby_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdladvanceblock_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlbreakaway_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlconcealbabyswall_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlmathold_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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
        let func = self.cached_function("pattern_row_cdlrisefall3methods_u8_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
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

        let matrix_len = rows.checked_mul(cols).ok_or_else(|| {
            CudaPatternRecognitionError::InvalidInput("rows*cols overflow".to_string())
        })?;
        if d_matrix.len() < matrix_len {
            return Err(CudaPatternRecognitionError::InvalidInput(
                "matrix buffer too small".to_string(),
            ));
        }

        let words_per_row = cols.div_ceil(64);
        let total_words = rows.checked_mul(words_per_row).ok_or_else(|| {
            CudaPatternRecognitionError::InvalidInput("rows*words overflow".to_string())
        })?;
        if d_words.len() < total_words {
            return Err(CudaPatternRecognitionError::InvalidInput(
                "words buffer too small".to_string(),
            ));
        }

        let func = self.cached_function("pattern_pack_u8_to_u64_kernel")?;

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
            self.launch_raw_function(func, grid, block, 0, args)?;
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

        let matrix_len = rows.checked_mul(cols).ok_or_else(|| {
            CudaPatternRecognitionError::InvalidInput("rows*cols overflow".to_string())
        })?;
        if matrix.len() != matrix_len {
            return Err(CudaPatternRecognitionError::InvalidInput(format!(
                "matrix length mismatch: expected {}, got {}",
                matrix_len,
                matrix.len()
            )));
        }

        let words_per_row = cols.div_ceil(64);
        let total_words = rows.checked_mul(words_per_row).ok_or_else(|| {
            CudaPatternRecognitionError::InvalidInput("rows*words overflow".to_string())
        })?;

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

        let len = rows.checked_mul(cols).ok_or_else(|| {
            CudaPatternRecognitionError::InvalidInput("rows*cols overflow".to_string())
        })?;
        if d_matrix_u8.len() < len {
            return Err(CudaPatternRecognitionError::InvalidInput(
                "input matrix buffer too small".to_string(),
            ));
        }

        let mut out = unsafe { DeviceBuffer::<f32>::uninitialized(len) }?;
        let func = self.cached_function("pattern_u8_to_f32_kernel")?;
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
            self.launch_raw_function(func, grid, block, 0, args)?;
        }

        Ok(DeviceArrayF32 {
            buf: out,
            rows,
            cols,
        })
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

        let len = rows.checked_mul(cols).ok_or_else(|| {
            CudaPatternRecognitionError::InvalidInput("rows*cols overflow".to_string())
        })?;

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
        extract_pattern_series, list_patterns, pattern_recognition_with_kernel,
        PatternRecognitionInput,
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
        let got = cuda
            .pack_matrix_u8_host(matrix.as_slice(), rows, cols)
            .unwrap();

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
