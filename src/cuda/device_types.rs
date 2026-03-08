#![cfg(feature = "cuda")]

use cust::context::Context;
use cust::memory::{DeviceBuffer, DeviceCopy};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum CudaDeviceViewError {
    #[error("null device pointer with non-zero length")]
    NullPointerWithNonZeroLength,
    #[error("matrix element count overflow")]
    MatrixLenOverflow,
    #[error("length mismatch for {name}: expected {expected}, actual {actual}")]
    LengthMismatch {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("device mismatch for {name}: expected {expected}, actual {actual}")]
    DeviceMismatch {
        name: &'static str,
        expected: u32,
        actual: u32,
    },
}

fn validate_raw_slice(device_ptr: u64, len: usize) -> Result<(), CudaDeviceViewError> {
    if len > 0 && device_ptr == 0 {
        return Err(CudaDeviceViewError::NullPointerWithNonZeroLength);
    }
    Ok(())
}

fn validate_matrix_len(rows: usize, cols: usize) -> Result<usize, CudaDeviceViewError> {
    rows.checked_mul(cols)
        .ok_or(CudaDeviceViewError::MatrixLenOverflow)
}

pub fn ensure_same_len(
    name: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), CudaDeviceViewError> {
    if expected != actual {
        return Err(CudaDeviceViewError::LengthMismatch {
            name,
            expected,
            actual,
        });
    }
    Ok(())
}

pub fn ensure_same_device(
    name: &'static str,
    expected: u32,
    actual: u32,
) -> Result<(), CudaDeviceViewError> {
    if expected != actual {
        return Err(CudaDeviceViewError::DeviceMismatch {
            name,
            expected,
            actual,
        });
    }
    Ok(())
}

macro_rules! define_device_slice_ref {
    ($name:ident, $ty:ty) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $name {
            device_ptr: u64,
            len: usize,
            device_id: u32,
            _marker: std::marker::PhantomData<$ty>,
        }

        impl $name {
            pub unsafe fn from_raw_parts(
                device_ptr: u64,
                len: usize,
                device_id: u32,
            ) -> Result<Self, CudaDeviceViewError> {
                validate_raw_slice(device_ptr, len)?;
                Ok(Self {
                    device_ptr,
                    len,
                    device_id,
                    _marker: std::marker::PhantomData,
                })
            }

            #[inline]
            pub fn device_ptr(&self) -> u64 {
                self.device_ptr
            }

            #[inline]
            pub fn len(&self) -> usize {
                self.len
            }

            #[inline]
            pub fn is_empty(&self) -> bool {
                self.len == 0
            }

            #[inline]
            pub fn device_id(&self) -> u32 {
                self.device_id
            }
        }
    };
}

define_device_slice_ref!(CudaDeviceSliceF32Ref, f32);
define_device_slice_ref!(CudaDeviceSliceI32Ref, i32);
define_device_slice_ref!(CudaDeviceSliceI64Ref, i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaDeviceMatrixF32Ref {
    device_ptr: u64,
    rows: usize,
    cols: usize,
    device_id: u32,
}

impl CudaDeviceMatrixF32Ref {
    pub unsafe fn from_raw_parts(
        device_ptr: u64,
        rows: usize,
        cols: usize,
        device_id: u32,
    ) -> Result<Self, CudaDeviceViewError> {
        let len = validate_matrix_len(rows, cols)?;
        validate_raw_slice(device_ptr, len)?;
        Ok(Self {
            device_ptr,
            rows,
            cols,
            device_id,
        })
    }

    #[inline]
    pub fn device_ptr(&self) -> u64 {
        self.device_ptr
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.rows.saturating_mul(self.cols)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }
}

pub struct CudaDeviceVector<T: DeviceCopy> {
    buf: DeviceBuffer<T>,
    len: usize,
    context: Arc<Context>,
    device_id: u32,
}

impl<T: DeviceCopy> std::fmt::Debug for CudaDeviceVector<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaDeviceVector")
            .field("device_ptr", &self.buf.as_device_ptr().as_raw())
            .field("len", &self.len)
            .field("device_id", &self.device_id)
            .finish()
    }
}

impl<T: DeviceCopy> CudaDeviceVector<T> {
    pub(crate) fn from_buffer(
        buf: DeviceBuffer<T>,
        len: usize,
        context: Arc<Context>,
        device_id: u32,
    ) -> Self {
        Self {
            buf,
            len,
            context,
            device_id,
        }
    }

    #[inline]
    pub fn buffer(&self) -> &DeviceBuffer<T> {
        &self.buf
    }

    #[inline]
    pub fn into_buffer(self) -> DeviceBuffer<T> {
        self.buf
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> {
        self.context.clone()
    }
}

pub type CudaDeviceVectorF32 = CudaDeviceVector<f32>;
pub type CudaDeviceVectorI32 = CudaDeviceVector<i32>;
pub type CudaDeviceVectorI64 = CudaDeviceVector<i64>;

impl CudaDeviceVector<f32> {
    #[inline]
    pub fn device_ptr(&self) -> u64 {
        self.buf.as_device_ptr().as_raw() as u64
    }

    pub fn as_view(&self) -> CudaDeviceSliceF32Ref {
        CudaDeviceSliceF32Ref {
            device_ptr: self.device_ptr(),
            len: self.len,
            device_id: self.device_id,
            _marker: std::marker::PhantomData,
        }
    }
}

impl CudaDeviceVector<i32> {
    #[inline]
    pub fn device_ptr(&self) -> u64 {
        self.buf.as_device_ptr().as_raw() as u64
    }

    pub fn as_view(&self) -> CudaDeviceSliceI32Ref {
        CudaDeviceSliceI32Ref {
            device_ptr: self.device_ptr(),
            len: self.len,
            device_id: self.device_id,
            _marker: std::marker::PhantomData,
        }
    }
}

impl CudaDeviceVector<i64> {
    #[inline]
    pub fn device_ptr(&self) -> u64 {
        self.buf.as_device_ptr().as_raw() as u64
    }

    pub fn as_view(&self) -> CudaDeviceSliceI64Ref {
        CudaDeviceSliceI64Ref {
            device_ptr: self.device_ptr(),
            len: self.len,
            device_id: self.device_id,
            _marker: std::marker::PhantomData,
        }
    }
}

pub struct CudaDeviceMatrix<T: DeviceCopy> {
    buf: DeviceBuffer<T>,
    rows: usize,
    cols: usize,
    context: Arc<Context>,
    device_id: u32,
}

impl<T: DeviceCopy> std::fmt::Debug for CudaDeviceMatrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaDeviceMatrix")
            .field("device_ptr", &self.buf.as_device_ptr().as_raw())
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("device_id", &self.device_id)
            .finish()
    }
}

impl<T: DeviceCopy> CudaDeviceMatrix<T> {
    pub(crate) fn from_buffer(
        buf: DeviceBuffer<T>,
        rows: usize,
        cols: usize,
        context: Arc<Context>,
        device_id: u32,
    ) -> Result<Self, CudaDeviceViewError> {
        let len = validate_matrix_len(rows, cols)?;
        ensure_same_len("matrix", len, buf.len())?;
        Ok(Self {
            buf,
            rows,
            cols,
            context,
            device_id,
        })
    }

    #[inline]
    pub fn buffer(&self) -> &DeviceBuffer<T> {
        &self.buf
    }

    #[inline]
    pub fn into_buffer(self) -> DeviceBuffer<T> {
        self.buf
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.rows.saturating_mul(self.cols)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> {
        self.context.clone()
    }
}

pub type CudaDeviceMatrixF32 = CudaDeviceMatrix<f32>;

impl CudaDeviceMatrix<f32> {
    #[inline]
    pub fn device_ptr(&self) -> u64 {
        self.buf.as_device_ptr().as_raw() as u64
    }

    pub fn as_view(&self) -> CudaDeviceMatrixF32Ref {
        CudaDeviceMatrixF32Ref {
            device_ptr: self.device_ptr(),
            rows: self.rows,
            cols: self.cols,
            device_id: self.device_id,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaDeviceOhlcRef {
    open: CudaDeviceSliceF32Ref,
    high: CudaDeviceSliceF32Ref,
    low: CudaDeviceSliceF32Ref,
    close: CudaDeviceSliceF32Ref,
    source: Option<CudaDeviceSliceF32Ref>,
}

impl CudaDeviceOhlcRef {
    pub fn new(
        open: CudaDeviceSliceF32Ref,
        high: CudaDeviceSliceF32Ref,
        low: CudaDeviceSliceF32Ref,
        close: CudaDeviceSliceF32Ref,
        source: Option<CudaDeviceSliceF32Ref>,
    ) -> Result<Self, CudaDeviceViewError> {
        let len = close.len();
        let device_id = close.device_id();
        ensure_same_len("ohlc.open", len, open.len())?;
        ensure_same_len("ohlc.high", len, high.len())?;
        ensure_same_len("ohlc.low", len, low.len())?;
        ensure_same_device("ohlc.open", device_id, open.device_id())?;
        ensure_same_device("ohlc.high", device_id, high.device_id())?;
        ensure_same_device("ohlc.low", device_id, low.device_id())?;
        if let Some(src) = source {
            ensure_same_len("ohlc.source", len, src.len())?;
            ensure_same_device("ohlc.source", device_id, src.device_id())?;
        }
        Ok(Self {
            open,
            high,
            low,
            close,
            source,
        })
    }

    #[inline]
    pub fn open(&self) -> CudaDeviceSliceF32Ref {
        self.open
    }

    #[inline]
    pub fn high(&self) -> CudaDeviceSliceF32Ref {
        self.high
    }

    #[inline]
    pub fn low(&self) -> CudaDeviceSliceF32Ref {
        self.low
    }

    #[inline]
    pub fn close(&self) -> CudaDeviceSliceF32Ref {
        self.close
    }

    #[inline]
    pub fn source(&self) -> Option<CudaDeviceSliceF32Ref> {
        self.source
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.close.len()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.close.device_id()
    }
}

#[derive(Debug)]
pub struct CudaDeviceOhlc {
    pub open: CudaDeviceVectorF32,
    pub high: CudaDeviceVectorF32,
    pub low: CudaDeviceVectorF32,
    pub close: CudaDeviceVectorF32,
    pub source: Option<CudaDeviceVectorF32>,
}

impl CudaDeviceOhlc {
    pub fn new(
        open: CudaDeviceVectorF32,
        high: CudaDeviceVectorF32,
        low: CudaDeviceVectorF32,
        close: CudaDeviceVectorF32,
        source: Option<CudaDeviceVectorF32>,
    ) -> Result<Self, CudaDeviceViewError> {
        let len = close.len();
        let device_id = close.device_id();
        ensure_same_len("ohlc.open", len, open.len())?;
        ensure_same_len("ohlc.high", len, high.len())?;
        ensure_same_len("ohlc.low", len, low.len())?;
        ensure_same_device("ohlc.open", device_id, open.device_id())?;
        ensure_same_device("ohlc.high", device_id, high.device_id())?;
        ensure_same_device("ohlc.low", device_id, low.device_id())?;
        if let Some(src) = source.as_ref() {
            ensure_same_len("ohlc.source", len, src.len())?;
            ensure_same_device("ohlc.source", device_id, src.device_id())?;
        }
        Ok(Self {
            open,
            high,
            low,
            close,
            source,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.close.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }

    pub fn as_view(&self) -> CudaDeviceOhlcRef {
        CudaDeviceOhlcRef {
            open: self.open.as_view(),
            high: self.high.as_view(),
            low: self.low.as_view(),
            close: self.close.as_view(),
            source: self.source.as_ref().map(|src| src.as_view()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaDeviceOhlcvRef {
    timestamp: Option<CudaDeviceSliceI64Ref>,
    open: CudaDeviceSliceF32Ref,
    high: CudaDeviceSliceF32Ref,
    low: CudaDeviceSliceF32Ref,
    close: CudaDeviceSliceF32Ref,
    volume: CudaDeviceSliceF32Ref,
    source: Option<CudaDeviceSliceF32Ref>,
}

impl CudaDeviceOhlcvRef {
    pub fn new(
        timestamp: Option<CudaDeviceSliceI64Ref>,
        open: CudaDeviceSliceF32Ref,
        high: CudaDeviceSliceF32Ref,
        low: CudaDeviceSliceF32Ref,
        close: CudaDeviceSliceF32Ref,
        volume: CudaDeviceSliceF32Ref,
        source: Option<CudaDeviceSliceF32Ref>,
    ) -> Result<Self, CudaDeviceViewError> {
        let len = close.len();
        let device_id = close.device_id();
        ensure_same_len("ohlcv.open", len, open.len())?;
        ensure_same_len("ohlcv.high", len, high.len())?;
        ensure_same_len("ohlcv.low", len, low.len())?;
        ensure_same_len("ohlcv.volume", len, volume.len())?;
        ensure_same_device("ohlcv.open", device_id, open.device_id())?;
        ensure_same_device("ohlcv.high", device_id, high.device_id())?;
        ensure_same_device("ohlcv.low", device_id, low.device_id())?;
        ensure_same_device("ohlcv.volume", device_id, volume.device_id())?;
        if let Some(ts) = timestamp {
            ensure_same_len("ohlcv.timestamp", len, ts.len())?;
            ensure_same_device("ohlcv.timestamp", device_id, ts.device_id())?;
        }
        if let Some(src) = source {
            ensure_same_len("ohlcv.source", len, src.len())?;
            ensure_same_device("ohlcv.source", device_id, src.device_id())?;
        }
        Ok(Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            source,
        })
    }

    #[inline]
    pub fn timestamp(&self) -> Option<CudaDeviceSliceI64Ref> {
        self.timestamp
    }

    #[inline]
    pub fn open(&self) -> CudaDeviceSliceF32Ref {
        self.open
    }

    #[inline]
    pub fn high(&self) -> CudaDeviceSliceF32Ref {
        self.high
    }

    #[inline]
    pub fn low(&self) -> CudaDeviceSliceF32Ref {
        self.low
    }

    #[inline]
    pub fn close(&self) -> CudaDeviceSliceF32Ref {
        self.close
    }

    #[inline]
    pub fn volume(&self) -> CudaDeviceSliceF32Ref {
        self.volume
    }

    #[inline]
    pub fn source(&self) -> Option<CudaDeviceSliceF32Ref> {
        self.source
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.close.len()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.close.device_id()
    }
}

#[derive(Debug)]
pub struct CudaDeviceOhlcv {
    pub timestamp: Option<CudaDeviceVectorI64>,
    pub open: CudaDeviceVectorF32,
    pub high: CudaDeviceVectorF32,
    pub low: CudaDeviceVectorF32,
    pub close: CudaDeviceVectorF32,
    pub volume: CudaDeviceVectorF32,
    pub source: Option<CudaDeviceVectorF32>,
}

impl CudaDeviceOhlcv {
    pub fn new(
        timestamp: Option<CudaDeviceVectorI64>,
        open: CudaDeviceVectorF32,
        high: CudaDeviceVectorF32,
        low: CudaDeviceVectorF32,
        close: CudaDeviceVectorF32,
        volume: CudaDeviceVectorF32,
        source: Option<CudaDeviceVectorF32>,
    ) -> Result<Self, CudaDeviceViewError> {
        let len = close.len();
        let device_id = close.device_id();
        ensure_same_len("ohlcv.open", len, open.len())?;
        ensure_same_len("ohlcv.high", len, high.len())?;
        ensure_same_len("ohlcv.low", len, low.len())?;
        ensure_same_len("ohlcv.volume", len, volume.len())?;
        ensure_same_device("ohlcv.open", device_id, open.device_id())?;
        ensure_same_device("ohlcv.high", device_id, high.device_id())?;
        ensure_same_device("ohlcv.low", device_id, low.device_id())?;
        ensure_same_device("ohlcv.volume", device_id, volume.device_id())?;
        if let Some(ts) = timestamp.as_ref() {
            ensure_same_len("ohlcv.timestamp", len, ts.len())?;
            ensure_same_device("ohlcv.timestamp", device_id, ts.device_id())?;
        }
        if let Some(src) = source.as_ref() {
            ensure_same_len("ohlcv.source", len, src.len())?;
            ensure_same_device("ohlcv.source", device_id, src.device_id())?;
        }
        Ok(Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            source,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.close.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }

    pub fn as_view(&self) -> CudaDeviceOhlcvRef {
        CudaDeviceOhlcvRef {
            timestamp: self.timestamp.as_ref().map(|ts| ts.as_view()),
            open: self.open.as_view(),
            high: self.high.as_view(),
            low: self.low.as_view(),
            close: self.close.as_view(),
            volume: self.volume.as_view(),
            source: self.source.as_ref().map(|src| src.as_view()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaDeviceHighLowRef {
    high: CudaDeviceSliceF32Ref,
    low: CudaDeviceSliceF32Ref,
}

impl CudaDeviceHighLowRef {
    pub fn new(
        high: CudaDeviceSliceF32Ref,
        low: CudaDeviceSliceF32Ref,
    ) -> Result<Self, CudaDeviceViewError> {
        ensure_same_len("high_low.low", high.len(), low.len())?;
        ensure_same_device("high_low.low", high.device_id(), low.device_id())?;
        Ok(Self { high, low })
    }

    #[inline]
    pub fn high(&self) -> CudaDeviceSliceF32Ref {
        self.high
    }

    #[inline]
    pub fn low(&self) -> CudaDeviceSliceF32Ref {
        self.low
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.high.len()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.high.device_id()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaDeviceCloseVolumeRef {
    close: CudaDeviceSliceF32Ref,
    volume: CudaDeviceSliceF32Ref,
}

impl CudaDeviceCloseVolumeRef {
    pub fn new(
        close: CudaDeviceSliceF32Ref,
        volume: CudaDeviceSliceF32Ref,
    ) -> Result<Self, CudaDeviceViewError> {
        ensure_same_len("close_volume.volume", close.len(), volume.len())?;
        ensure_same_device("close_volume.volume", close.device_id(), volume.device_id())?;
        Ok(Self { close, volume })
    }

    #[inline]
    pub fn close(&self) -> CudaDeviceSliceF32Ref {
        self.close
    }

    #[inline]
    pub fn volume(&self) -> CudaDeviceSliceF32Ref {
        self.volume
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.close.len()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.close.device_id()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_view_rejects_null_pointer_with_non_zero_len() {
        let err = unsafe { CudaDeviceSliceF32Ref::from_raw_parts(0, 4, 0) }.unwrap_err();
        assert_eq!(err, CudaDeviceViewError::NullPointerWithNonZeroLength);
    }

    #[test]
    fn ensure_same_device_reports_mismatch() {
        let err = ensure_same_device("prices", 0, 1).unwrap_err();
        assert_eq!(
            err,
            CudaDeviceViewError::DeviceMismatch {
                name: "prices",
                expected: 0,
                actual: 1,
            }
        );
    }

    #[test]
    fn ohlc_ref_requires_matching_lengths() {
        let open = unsafe { CudaDeviceSliceF32Ref::from_raw_parts(0x1000, 8, 0) }.unwrap();
        let high = unsafe { CudaDeviceSliceF32Ref::from_raw_parts(0x2000, 8, 0) }.unwrap();
        let low = unsafe { CudaDeviceSliceF32Ref::from_raw_parts(0x3000, 7, 0) }.unwrap();
        let close = unsafe { CudaDeviceSliceF32Ref::from_raw_parts(0x4000, 8, 0) }.unwrap();

        let err = CudaDeviceOhlcRef::new(open, high, low, close, None).unwrap_err();
        assert_eq!(
            err,
            CudaDeviceViewError::LengthMismatch {
                name: "ohlc.low",
                expected: 8,
                actual: 7,
            }
        );
    }
}
