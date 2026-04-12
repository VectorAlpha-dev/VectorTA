#![cfg(feature = "cuda")]

use super::device_types::{
    ensure_same_device, CudaDeviceCloseVolumeRef, CudaDeviceHighLowRef, CudaDeviceMatrixF32,
    CudaDeviceOhlc, CudaDeviceOhlcv, CudaDeviceVectorF32, CudaDeviceVectorI32, CudaDeviceVectorI64,
    CudaDeviceViewError,
};
use cust::context::Context;
use cust::device::Device;
use cust::error::CudaError;
use cust::memory::{CopyDestination, DeviceBuffer};
use cust::prelude::CudaFlags;
use cust::stream::{Stream, StreamFlags};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaRuntimeError {
    #[error(transparent)]
    Cuda(#[from] CudaError),
    #[error(transparent)]
    View(#[from] CudaDeviceViewError),
}

#[derive(Clone)]
pub struct CudaSession {
    context: Arc<Context>,
    stream: Arc<Stream>,
    device_id: u32,
}

impl std::fmt::Debug for CudaSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaSession")
            .field("device_id", &self.device_id)
            .finish()
    }
}

impl CudaSession {
    pub fn new(device_id: usize) -> Result<Self, CudaRuntimeError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let stream = Arc::new(Stream::new(StreamFlags::NON_BLOCKING, None)?);
        Ok(Self {
            context,
            stream,
            device_id: device_id as u32,
        })
    }

    #[inline]
    pub fn from_parts(context: Arc<Context>, stream: Arc<Stream>, device_id: u32) -> Self {
        Self {
            context,
            stream,
            device_id,
        }
    }

    #[inline]
    pub fn stream(&self) -> &Stream {
        self.stream.as_ref()
    }

    #[inline]
    pub fn stream_arc(&self) -> Arc<Stream> {
        self.stream.clone()
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> {
        self.context.clone()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    pub fn synchronize(&self) -> Result<(), CudaRuntimeError> {
        self.stream.synchronize()?;
        Ok(())
    }
}

pub struct CudaRuntime {
    session: Arc<CudaSession>,
    #[cfg(test)]
    _test_lock: super::CudaTestLock,
}

impl std::fmt::Debug for CudaRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaRuntime")
            .field("device_id", &self.session.device_id())
            .finish()
    }
}

impl CudaRuntime {
    pub fn new(device_id: usize) -> Result<Self, CudaRuntimeError> {
        #[cfg(test)]
        let test_lock = super::cuda_test_lock();

        Ok(Self {
            session: Arc::new(CudaSession::new(device_id)?),
            #[cfg(test)]
            _test_lock: test_lock,
        })
    }

    #[inline]
    pub fn stream(&self) -> &Stream {
        self.session.stream()
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> {
        self.session.context_arc()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.session.device_id()
    }

    #[inline]
    pub fn session_arc(&self) -> Arc<CudaSession> {
        self.session.clone()
    }

    pub fn synchronize(&self) -> Result<(), CudaRuntimeError> {
        self.session.synchronize()
    }

    pub fn upload_f32(&self, values: &[f32]) -> Result<CudaDeviceVectorF32, CudaRuntimeError> {
        let buf = DeviceBuffer::from_slice(values)?;
        Ok(CudaDeviceVectorF32::from_buffer(
            buf,
            values.len(),
            self.context_arc(),
            self.device_id(),
        ))
    }

    pub fn upload_i32(&self, values: &[i32]) -> Result<CudaDeviceVectorI32, CudaRuntimeError> {
        let buf = DeviceBuffer::from_slice(values)?;
        Ok(CudaDeviceVectorI32::from_buffer(
            buf,
            values.len(),
            self.context_arc(),
            self.device_id(),
        ))
    }

    pub fn upload_i64(&self, values: &[i64]) -> Result<CudaDeviceVectorI64, CudaRuntimeError> {
        let buf = DeviceBuffer::from_slice(values)?;
        Ok(CudaDeviceVectorI64::from_buffer(
            buf,
            values.len(),
            self.context_arc(),
            self.device_id(),
        ))
    }

    pub fn upload_matrix_f32(
        &self,
        values: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<CudaDeviceMatrixF32, CudaRuntimeError> {
        let buf = DeviceBuffer::from_slice(values)?;
        Ok(CudaDeviceMatrixF32::from_buffer(
            buf,
            rows,
            cols,
            self.context_arc(),
            self.device_id(),
        )?)
    }

    pub fn upload_ohlc(
        &self,
        open: &[f32],
        high: &[f32],
        low: &[f32],
        close: &[f32],
        source: Option<&[f32]>,
    ) -> Result<CudaDeviceOhlc, CudaRuntimeError> {
        let open = self.upload_f32(open)?;
        let high = self.upload_f32(high)?;
        let low = self.upload_f32(low)?;
        let close = self.upload_f32(close)?;
        let source = match source {
            Some(values) => Some(self.upload_f32(values)?),
            None => None,
        };
        Ok(CudaDeviceOhlc::new(open, high, low, close, source)?)
    }

    pub fn upload_ohlcv(
        &self,
        timestamp: Option<&[i64]>,
        open: &[f32],
        high: &[f32],
        low: &[f32],
        close: &[f32],
        volume: &[f32],
        source: Option<&[f32]>,
    ) -> Result<CudaDeviceOhlcv, CudaRuntimeError> {
        let timestamp = match timestamp {
            Some(values) => Some(self.upload_i64(values)?),
            None => None,
        };
        let open = self.upload_f32(open)?;
        let high = self.upload_f32(high)?;
        let low = self.upload_f32(low)?;
        let close = self.upload_f32(close)?;
        let volume = self.upload_f32(volume)?;
        let source = match source {
            Some(values) => Some(self.upload_f32(values)?),
            None => None,
        };
        Ok(CudaDeviceOhlcv::new(
            timestamp, open, high, low, close, volume, source,
        )?)
    }

    pub fn download_f32(&self, values: &CudaDeviceVectorF32) -> Result<Vec<f32>, CudaRuntimeError> {
        ensure_same_device("runtime.download_f32", self.device_id(), values.device_id())?;
        let mut host = vec![0.0f32; values.len()];
        values.buffer().copy_to(host.as_mut_slice())?;
        Ok(host)
    }

    pub fn download_matrix_f32(
        &self,
        values: &CudaDeviceMatrixF32,
    ) -> Result<Vec<f32>, CudaRuntimeError> {
        ensure_same_device(
            "runtime.download_matrix_f32",
            self.device_id(),
            values.device_id(),
        )?;
        let mut host = vec![0.0f32; values.len()];
        values.buffer().copy_to(host.as_mut_slice())?;
        Ok(host)
    }

    pub fn validate_high_low(&self, data: CudaDeviceHighLowRef) -> Result<(), CudaRuntimeError> {
        ensure_same_device("runtime.high_low", self.device_id(), data.device_id())?;
        Ok(())
    }

    pub fn validate_close_volume(
        &self,
        data: CudaDeviceCloseVolumeRef,
    ) -> Result<(), CudaRuntimeError> {
        ensure_same_device("runtime.close_volume", self.device_id(), data.device_id())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::moving_averages::ma_selector::{CudaMaDeviceDataRef, CudaMaSelector};
    use crate::cuda::moving_averages::CudaOtt;
    use crate::indicators::ott::OttBatchRange;

    #[test]
    fn runtime_roundtrip_upload_download_f32_when_cuda_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let runtime = CudaRuntime::new(0).expect("runtime");
        let values = [1.0f32, 2.5, -3.0, 4.25];
        let dev = runtime.upload_f32(&values).expect("upload");
        let host = runtime.download_f32(&dev).expect("download");
        assert_eq!(host, values);
    }

    #[test]
    fn runtime_roundtrip_upload_download_matrix_when_cuda_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let runtime = CudaRuntime::new(0).expect("runtime");
        let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let dev = runtime
            .upload_matrix_f32(&values, 2, 3)
            .expect("upload matrix");
        let host = runtime.download_matrix_f32(&dev).expect("download matrix");
        assert_eq!(host, values);
    }

    #[test]
    fn runtime_validates_matching_refs_when_cuda_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let runtime = CudaRuntime::new(0).expect("runtime");
        let high = runtime.upload_f32(&[1.0f32, 2.0, 3.0]).expect("high");
        let low = runtime.upload_f32(&[0.5f32, 1.5, 2.5]).expect("low");
        let close = runtime.upload_f32(&[1.1f32, 2.1, 3.1]).expect("close");
        let volume = runtime.upload_f32(&[10.0f32, 11.0, 12.0]).expect("volume");

        let high_low = CudaDeviceHighLowRef::new(high.as_view(), low.as_view()).expect("high_low");
        let close_volume =
            CudaDeviceCloseVolumeRef::new(close.as_view(), volume.as_view()).expect("close_volume");
        runtime
            .validate_high_low(high_low)
            .expect("validate high_low");
        runtime
            .validate_close_volume(close_volume)
            .expect("validate close_volume");
    }

    #[test]
    fn runtime_uploads_ohlc_and_ohlcv_when_cuda_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let runtime = CudaRuntime::new(0).expect("runtime");
        let ohlc = runtime
            .upload_ohlc(
                &[1.0f32, 2.0, 3.0],
                &[2.0f32, 3.0, 4.0],
                &[0.5f32, 1.5, 2.5],
                &[1.5f32, 2.5, 3.5],
                None,
            )
            .expect("upload ohlc");
        assert_eq!(ohlc.len(), 3);
        assert_eq!(ohlc.as_view().device_id(), runtime.device_id());

        let ohlcv = runtime
            .upload_ohlcv(
                Some(&[1i64, 2, 3]),
                &[1.0f32, 2.0, 3.0],
                &[2.0f32, 3.0, 4.0],
                &[0.5f32, 1.5, 2.5],
                &[1.5f32, 2.5, 3.5],
                &[10.0f32, 11.0, 12.0],
                None,
            )
            .expect("upload ohlcv");
        assert_eq!(ohlcv.len(), 3);
        assert_eq!(ohlcv.as_view().device_id(), runtime.device_id());
    }

    #[test]
    fn runtime_session_supports_chained_selector_and_ott_when_cuda_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let runtime = CudaRuntime::new(0).expect("runtime");
        let session = runtime.session_arc();
        let mut prices = vec![f32::NAN; 256];
        for (i, value) in prices.iter_mut().enumerate().skip(4) {
            let x = i as f32;
            *value = (x * 0.013).sin() + 0.0015 * x;
        }
        let d_prices = runtime.upload_f32(&prices).expect("upload");

        let selector = CudaMaSelector::from_session(session.clone());
        let ma_rows = selector
            .device_native()
            .ma_sweep_to_device_ref(
                "ema",
                CudaMaDeviceDataRef::Slice(d_prices.as_view()),
                4,
                8,
                12,
                2,
            )
            .expect("selector on shared session");
        assert_eq!(ma_rows.rows, 3);
        assert_eq!(ma_rows.cols, prices.len());

        let sweep = OttBatchRange {
            period: (8, 12, 2),
            percent: (1.0, 1.4, 0.4),
            ma_types: vec!["EMA".to_string()],
        };
        let shared = CudaOtt::from_session(session)
            .expect("shared-session ott")
            .ott_batch_dev_from_device_prices(d_prices.buffer(), prices.len(), 4, &sweep)
            .expect("shared-session ott batch");
        let fresh = CudaOtt::new(0)
            .expect("fresh ott")
            .ott_batch_dev_from_device_prices(d_prices.buffer(), prices.len(), 4, &sweep)
            .expect("fresh-session ott batch");

        let mut shared_host = vec![0f32; shared.rows * shared.cols];
        shared
            .buf
            .copy_to(shared_host.as_mut_slice())
            .expect("copy shared");
        let mut fresh_host = vec![0f32; fresh.rows * fresh.cols];
        fresh
            .buf
            .copy_to(fresh_host.as_mut_slice())
            .expect("copy fresh");

        assert_eq!(shared.rows, fresh.rows);
        assert_eq!(shared.cols, fresh.cols);
        for (idx, (&lhs, &rhs)) in shared_host.iter().zip(fresh_host.iter()).enumerate() {
            if lhs.is_nan() && rhs.is_nan() {
                continue;
            }
            assert!(
                (lhs - rhs).abs() <= 1e-5,
                "shared-session drift at {idx}: lhs={lhs} rhs={rhs}"
            );
        }
    }
}
