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

pub struct CudaRuntime {
    context: Arc<Context>,
    stream: Stream,
    device_id: u32,
}

impl std::fmt::Debug for CudaRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaRuntime")
            .field("device_id", &self.device_id)
            .finish()
    }
}

impl CudaRuntime {
    pub fn new(device_id: usize) -> Result<Self, CudaRuntimeError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            context,
            stream,
            device_id: device_id as u32,
        })
    }

    #[inline]
    pub fn stream(&self) -> &Stream {
        &self.stream
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

    pub fn upload_f32(&self, values: &[f32]) -> Result<CudaDeviceVectorF32, CudaRuntimeError> {
        let buf = DeviceBuffer::from_slice(values)?;
        Ok(CudaDeviceVectorF32::from_buffer(
            buf,
            values.len(),
            self.context.clone(),
            self.device_id,
        ))
    }

    pub fn upload_i32(&self, values: &[i32]) -> Result<CudaDeviceVectorI32, CudaRuntimeError> {
        let buf = DeviceBuffer::from_slice(values)?;
        Ok(CudaDeviceVectorI32::from_buffer(
            buf,
            values.len(),
            self.context.clone(),
            self.device_id,
        ))
    }

    pub fn upload_i64(&self, values: &[i64]) -> Result<CudaDeviceVectorI64, CudaRuntimeError> {
        let buf = DeviceBuffer::from_slice(values)?;
        Ok(CudaDeviceVectorI64::from_buffer(
            buf,
            values.len(),
            self.context.clone(),
            self.device_id,
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
            self.context.clone(),
            self.device_id,
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
        ensure_same_device("runtime.download_f32", self.device_id, values.device_id())?;
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
            self.device_id,
            values.device_id(),
        )?;
        let mut host = vec![0.0f32; values.len()];
        values.buffer().copy_to(host.as_mut_slice())?;
        Ok(host)
    }

    pub fn validate_high_low(&self, data: CudaDeviceHighLowRef) -> Result<(), CudaRuntimeError> {
        ensure_same_device("runtime.high_low", self.device_id, data.device_id())?;
        Ok(())
    }

    pub fn validate_close_volume(
        &self,
        data: CudaDeviceCloseVolumeRef,
    ) -> Result<(), CudaRuntimeError> {
        ensure_same_device("runtime.close_volume", self.device_id, data.device_id())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
