use cudarc::driver::CudaDevice;
use std::sync::Arc;
use std::error::Error;
use std::fmt;

use crate::indicators::moving_averages::alma::{AlmaError, AlmaBatchOutput, AlmaBatchRange, AlmaParams};

#[derive(Debug)]
pub enum CudaAlmaError {
    CudaError(String),
    AlmaError(AlmaError),
    InvalidInput(String),
    NotImplemented,
}

impl fmt::Display for CudaAlmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaAlmaError::CudaError(e) => write!(f, "CUDA error: {}", e),
            CudaAlmaError::AlmaError(e) => write!(f, "ALMA error: {}", e),
            CudaAlmaError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
            CudaAlmaError::NotImplemented => write!(f, "CUDA implementation is not complete"),
        }
    }
}

impl Error for CudaAlmaError {}

impl From<AlmaError> for CudaAlmaError {
    fn from(e: AlmaError) -> Self {
        CudaAlmaError::AlmaError(e)
    }
}

pub struct CudaAlma {
    device: Arc<CudaDevice>,
}

impl CudaAlma {
    pub fn new(device_id: usize) -> Result<Self, CudaAlmaError> {
        let device = CudaDevice::new(device_id)
            .map_err(|e| CudaAlmaError::CudaError(e.to_string()))?;
        
        // TODO: Implement proper PTX loading once we understand the cudarc 0.12 API better
        // For now, this is a placeholder implementation
        
        Ok(Self {
            device,
        })
    }
    
    /// Process batch ALMA calculations on GPU
    pub fn alma_batch(&self,
        _data: &[f64],
        _sweep: &AlmaBatchRange,
        _first_valid: usize
    ) -> Result<AlmaBatchOutput, CudaAlmaError> {
        // TODO: Implement CUDA kernel launching
        Err(CudaAlmaError::NotImplemented)
    }
}

// Helper function to compute weights on CPU (for single calculations)
fn compute_weights_cpu(period: usize, offset: f64, sigma: f64) -> (Vec<f64>, f64) {
    let m = offset * (period - 1) as f64;
    let s = period as f64 / sigma;
    let s2 = 2.0 * s * s;
    
    let mut weights = Vec::with_capacity(period);
    let mut norm = 0.0;
    
    for i in 0..period {
        let diff = i as f64 - m;
        let w = (-(diff * diff) / s2).exp();
        weights.push(w);
        norm += w;
    }
    
    (weights, 1.0 / norm)
}

// Expand parameter grid (matching CPU implementation)
fn expand_grid(r: &AlmaBatchRange) -> Vec<AlmaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return vec![start];
        }
        let mut v = Vec::new();
        let mut x = start;
        while x <= end + 1e-12 {
            v.push(x);
            x += step;
        }
        v
    }
    
    let periods = axis_usize(r.period);
    let offsets = axis_f64(r.offset);
    let sigmas = axis_f64(r.sigma);
    
    let mut out = Vec::with_capacity(periods.len() * offsets.len() * sigmas.len());
    for &p in &periods {
        for &o in &offsets {
            for &s in &sigmas {
                out.push(AlmaParams {
                    period: Some(p),
                    offset: Some(o),
                    sigma: Some(s),
                });
            }
        }
    }
    out
}