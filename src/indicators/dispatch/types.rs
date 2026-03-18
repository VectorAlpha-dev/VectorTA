#[cfg(feature = "cuda")]
use crate::cuda::{
    CudaDeviceCloseVolumeRef, CudaDeviceHighLowRef, CudaDeviceOhlcRef, CudaDeviceOhlcvRef,
    CudaDeviceSliceF32Ref,
};
use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParamValue<'a> {
    Int(i64),
    Float(f64),
    Bool(bool),
    EnumString(&'a str),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParamKV<'a> {
    pub key: &'a str,
    pub value: ParamValue<'a>,
}

#[derive(Debug, Clone, Copy)]
pub enum IndicatorDataRef<'a> {
    Slice {
        values: &'a [f64],
    },
    Candles {
        candles: &'a Candles,
        source: Option<&'a str>,
    },
    Ohlc {
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
    Ohlcv {
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
    },
    HighLow {
        high: &'a [f64],
        low: &'a [f64],
    },
    CloseVolume {
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone, Copy)]
pub struct IndicatorParamSet<'a> {
    pub params: &'a [ParamKV<'a>],
}

#[derive(Debug, Clone, Copy)]
pub struct IndicatorBatchRequest<'a> {
    pub indicator_id: &'a str,
    pub output_id: Option<&'a str>,
    pub data: IndicatorDataRef<'a>,
    pub combos: &'a [IndicatorParamSet<'a>],
    pub kernel: Kernel,
}

#[derive(Debug, Clone, Default)]
pub struct IndicatorBatchOutput {
    pub output_id: String,
    pub rows: usize,
    pub cols: usize,
    pub values_f64: Option<Vec<f64>>,
    pub values_i32: Option<Vec<i32>>,
    pub values_bool: Option<Vec<bool>>,
}

#[derive(Debug, Clone, Copy)]
pub struct IndicatorComputeRequest<'a> {
    pub indicator_id: &'a str,
    pub output_id: Option<&'a str>,
    pub data: IndicatorDataRef<'a>,
    pub params: &'a [ParamKV<'a>],
    pub kernel: Kernel,
}

#[derive(Debug, Clone, PartialEq)]
pub enum IndicatorSeries {
    F64(Vec<f64>),
    I32(Vec<i32>),
    Bool(Vec<bool>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct IndicatorComputeOutput {
    pub output_id: String,
    pub series: IndicatorSeries,
    pub warmup: Option<usize>,
    pub rows: usize,
    pub cols: usize,
    pub pattern_ids: Option<Vec<String>>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaOutputTarget {
    DeviceF32,
    HostF32,
}

#[cfg(feature = "cuda")]
#[derive(Clone)]
pub struct DeviceMatrixF32 {
    pub device_ptr: u64,
    pub rows: usize,
    pub cols: usize,
    pub device_id: u32,
    owner: std::sync::Arc<crate::cuda::moving_averages::DeviceArrayF32>,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for DeviceMatrixF32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceMatrixF32")
            .field("device_ptr", &self.device_ptr)
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("device_id", &self.device_id)
            .finish()
    }
}

#[cfg(feature = "cuda")]
impl PartialEq for DeviceMatrixF32 {
    fn eq(&self, other: &Self) -> bool {
        self.device_ptr == other.device_ptr
            && self.rows == other.rows
            && self.cols == other.cols
            && self.device_id == other.device_id
    }
}

#[cfg(feature = "cuda")]
impl Eq for DeviceMatrixF32 {}

#[cfg(feature = "cuda")]
impl DeviceMatrixF32 {
    pub(crate) fn from_owned(
        owner: crate::cuda::moving_averages::DeviceArrayF32,
        device_id: u32,
    ) -> Self {
        let owner = std::sync::Arc::new(owner);
        Self {
            device_ptr: owner.device_ptr(),
            rows: owner.rows,
            cols: owner.cols,
            device_id,
            owner,
        }
    }

    pub fn len(&self) -> usize {
        self.rows.saturating_mul(self.cols)
    }

    pub fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }

    pub(crate) fn owner(&self) -> &crate::cuda::moving_averages::DeviceArrayF32 {
        &self.owner
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
pub enum IndicatorCudaDataRef<'a> {
    Slice {
        values: &'a [f32],
    },
    Ohlc {
        open: &'a [f32],
        high: &'a [f32],
        low: &'a [f32],
        close: &'a [f32],
        source: Option<&'a [f32]>,
    },
    Ohlcv {
        timestamp: Option<&'a [i64]>,
        open: &'a [f32],
        high: &'a [f32],
        low: &'a [f32],
        close: &'a [f32],
        volume: &'a [f32],
        source: Option<&'a [f32]>,
    },
    HighLow {
        high: &'a [f32],
        low: &'a [f32],
    },
    CloseVolume {
        close: &'a [f32],
        volume: &'a [f32],
    },
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
pub struct IndicatorCudaRequest<'a> {
    pub indicator_id: &'a str,
    pub output_id: Option<&'a str>,
    pub data: IndicatorCudaDataRef<'a>,
    pub params: &'a [ParamKV<'a>],
    pub kernel: Kernel,
    pub target: CudaOutputTarget,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
pub struct IndicatorCudaBitmaskRequest<'a> {
    pub indicator_id: &'a str,
    pub output_id: Option<&'a str>,
    pub data: IndicatorCudaDataRef<'a>,
    pub params: &'a [ParamKV<'a>],
    pub kernel: Kernel,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndicatorCudaDeviceDataRef {
    Slice { values: CudaDeviceSliceF32Ref },
    Ohlc(CudaDeviceOhlcRef),
    Ohlcv(CudaDeviceOhlcvRef),
    HighLow(CudaDeviceHighLowRef),
    CloseVolume(CudaDeviceCloseVolumeRef),
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
pub struct IndicatorCudaDeviceRequest<'a> {
    pub indicator_id: &'a str,
    pub output_id: Option<&'a str>,
    pub data: IndicatorCudaDeviceDataRef,
    pub params: &'a [ParamKV<'a>],
    pub kernel: Kernel,
    pub target: CudaOutputTarget,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
pub struct IndicatorCudaDeviceBitmaskRequest<'a> {
    pub indicator_id: &'a str,
    pub output_id: Option<&'a str>,
    pub data: IndicatorCudaDeviceDataRef,
    pub params: &'a [ParamKV<'a>],
    pub kernel: Kernel,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, PartialEq)]
pub enum IndicatorCudaSeries {
    DeviceF32(DeviceMatrixF32),
    HostF32(Vec<f32>),
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, PartialEq)]
pub struct IndicatorCudaOutput {
    pub output_id: String,
    pub series: IndicatorCudaSeries,
    pub warmup: Option<usize>,
    pub rows: usize,
    pub cols: usize,
    pub pattern_ids: Option<Vec<String>>,
}

#[cfg(feature = "cuda")]
pub struct PatternRecognitionCudaBitmaskOutput {
    pub output_id: String,
    pub series: crate::cuda::pattern_recognition_wrapper::DevicePatternBitmaskU64,
    pub warmup: Option<usize>,
    pub rows: usize,
    pub cols: usize,
    pub words_per_row: usize,
    pub pattern_ids: Vec<String>,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for PatternRecognitionCudaBitmaskOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PatternRecognitionCudaBitmaskOutput")
            .field("output_id", &self.output_id)
            .field("device_ptr", &self.series.device_ptr())
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("words_per_row", &self.words_per_row)
            .field("device_id", &self.series.device_id)
            .field("warmup", &self.warmup)
            .field("pattern_ids", &self.pattern_ids)
            .finish()
    }
}
