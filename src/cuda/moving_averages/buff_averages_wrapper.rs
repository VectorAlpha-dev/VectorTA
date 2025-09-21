//! CUDA scaffolding for Buff Averages kernels.
//!
//! Mirrors the ALMA integration: inputs are converted to FP32, masked prefix
//! sums are built on the host, and a lightweight CUDA kernel consumes the
//! prefix buffers to evaluate all fast/slow combinations in parallel.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::buff_averages::BuffAveragesBatchRange;
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaBuffAveragesError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaBuffAveragesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaBuffAveragesError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaBuffAveragesError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "buff_averages",
            "batch_dev",
            "buff_averages_cuda_batch_dev",
            "60k_x_49combos",
            prep_buff_averages_batch_box,
        )]
    }

    struct BuffAveragesBatchState {
        cuda: CudaBuffAverages,
        price: Vec<f32>,
        volume: Vec<f32>,
        sweep: BuffAveragesBatchRange,
    }

    impl CudaBenchState for BuffAveragesBatchState {
        fn launch(&mut self) {
            let (fast_dev, slow_dev) = self
                .cuda
                .buff_averages_batch_dev(&self.price, &self.volume, &self.sweep)
                .expect("launch buff averages batch kernel");
            drop(fast_dev);
            drop(slow_dev);
        }
    }

    fn prep_buff_averages_batch() -> BuffAveragesBatchState {
        let cuda = CudaBuffAverages::new(0).expect("cuda buff averages");
        let len = 60_000usize;
        let mut price = vec![f32::NAN; len];
        let mut volume = vec![f32::NAN; len];
        for i in 3..len {
            let x = i as f32;
            price[i] = (x * 0.001).sin() + 0.0001 * x;
            volume[i] = (x * 0.0007).cos().abs() + 0.6;
        }
        let sweep = BuffAveragesBatchRange {
            fast_period: (4, 28, 4),
            slow_period: (32, 128, 16),
        };
        BuffAveragesBatchState {
            cuda,
            price,
            volume,
            sweep,
        }
    }

    fn prep_buff_averages_batch_box() -> Box<dyn CudaBenchState> {
        Box::new(prep_buff_averages_batch())
    }
}

impl std::error::Error for CudaBuffAveragesError {}

pub struct CudaBuffAverages {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaBuffAverages {
    pub fn new(device_id: usize) -> Result<Self, CudaBuffAveragesError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        let context =
            Context::new(device).map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/buff_averages_kernel.ptx"));
        let module =
            Module::from_ptx(ptx, &[]).map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    fn expand_grid(range: &BuffAveragesBatchRange) -> Vec<(usize, usize)> {
        fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
            if step == 0 || start == end {
                return vec![start];
            }
            (start..=end).step_by(step).collect()
        }

        let fasts = axis(range.fast_period);
        let slows = axis(range.slow_period);
        let mut combos = Vec::with_capacity(fasts.len() * slows.len());
        for &fast in &fasts {
            for &slow in &slows {
                combos.push((fast, slow));
            }
        }
        combos
    }

    fn prepare_batch_inputs(
        price_f32: &[f32],
        volume_f32: &[f32],
        sweep: &BuffAveragesBatchRange,
    ) -> Result<(Vec<(usize, usize)>, usize, usize), CudaBuffAveragesError> {
        if price_f32.is_empty() {
            return Err(CudaBuffAveragesError::InvalidInput(
                "empty price data".into(),
            ));
        }
        if price_f32.len() != volume_f32.len() {
            return Err(CudaBuffAveragesError::InvalidInput(format!(
                "price/volume length mismatch ({} vs {})",
                price_f32.len(),
                volume_f32.len()
            )));
        }

        let len = price_f32.len();
        let first_valid = price_f32.iter().position(|v| !v.is_nan()).ok_or_else(|| {
            CudaBuffAveragesError::InvalidInput("all price values are NaN".into())
        })?;

        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaBuffAveragesError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        for &(fast, slow) in &combos {
            if fast == 0 || slow == 0 {
                return Err(CudaBuffAveragesError::InvalidInput(format!(
                    "invalid periods (fast={}, slow={})",
                    fast, slow
                )));
            }
            if fast > len || slow > len {
                return Err(CudaBuffAveragesError::InvalidInput(format!(
                    "period exceeds length (len={}, fast={}, slow={})",
                    len, fast, slow
                )));
            }
            if len - first_valid < slow {
                return Err(CudaBuffAveragesError::InvalidInput(format!(
                    "not enough valid data for slow={} (valid after first={}): {}",
                    slow,
                    first_valid,
                    len - first_valid
                )));
            }
            if fast > slow {
                return Err(CudaBuffAveragesError::InvalidInput(format!(
                    "fast period {} must be <= slow period {}",
                    fast, slow
                )));
            }
        }

        Ok((combos, first_valid, len))
    }

    fn build_prefix_sums(price: &[f32], volume: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let len = price.len();
        let mut prefix_pv = vec![0.0f32; len + 1];
        let mut prefix_vv = vec![0.0f32; len + 1];
        let mut acc_pv = 0.0f64;
        let mut acc_vv = 0.0f64;
        for i in 0..len {
            let p = price[i];
            let v = volume[i];
            let (pv, vv) = if p.is_nan() || v.is_nan() {
                (0.0f64, 0.0f64)
            } else {
                let pf = p as f64;
                let vf = v as f64;
                (pf * vf, vf)
            };
            acc_pv += pv;
            acc_vv += vv;
            prefix_pv[i + 1] = acc_pv as f32;
            prefix_vv[i + 1] = acc_vv as f32;
        }
        (prefix_pv, prefix_vv)
    }

    fn launch_batch_kernel(
        &self,
        d_prefix_pv: &DeviceBuffer<f32>,
        d_prefix_vv: &DeviceBuffer<f32>,
        d_fast: &DeviceBuffer<i32>,
        d_slow: &DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_fast_out: &mut DeviceBuffer<f32>,
        d_slow_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaBuffAveragesError> {
        let func = self
            .module
            .get_function("buff_averages_batch_prefix_f32")
            .map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;

        let block_x: u32 = 256;
        let grid_x = ((len as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), n_combos as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prefix_pv_ptr = d_prefix_pv.as_device_ptr().as_raw();
            let mut prefix_vv_ptr = d_prefix_vv.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut first_valid_i = first_valid as i32;
            let mut fast_ptr = d_fast.as_device_ptr().as_raw();
            let mut slow_ptr = d_slow.as_device_ptr().as_raw();
            let mut combos_i = n_combos as i32;
            let mut fast_out_ptr = d_fast_out.as_device_ptr().as_raw();
            let mut slow_out_ptr = d_slow_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prefix_pv_ptr as *mut _ as *mut c_void,
                &mut prefix_vv_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut fast_ptr as *mut _ as *mut c_void,
                &mut slow_ptr as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut fast_out_ptr as *mut _ as *mut c_void,
                &mut slow_out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    fn run_batch_kernel(
        &self,
        price_f32: &[f32],
        volume_f32: &[f32],
        combos: &[(usize, usize)],
        first_valid: usize,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32), CudaBuffAveragesError> {
        let len = price_f32.len();
        let (prefix_pv, prefix_vv) = Self::build_prefix_sums(price_f32, volume_f32);

        let d_prefix_pv = DeviceBuffer::from_slice(&prefix_pv)
            .map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        let d_prefix_vv = DeviceBuffer::from_slice(&prefix_vv)
            .map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;

        let fast_periods: Vec<i32> = combos.iter().map(|&(f, _)| f as i32).collect();
        let slow_periods: Vec<i32> = combos.iter().map(|&(_, s)| s as i32).collect();
        let d_fast = DeviceBuffer::from_slice(&fast_periods)
            .map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        let d_slow = DeviceBuffer::from_slice(&slow_periods)
            .map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;

        let elems = combos.len() * len;
        let mut d_fast_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }
            .map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        let mut d_slow_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }
            .map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prefix_pv,
            &d_prefix_vv,
            &d_fast,
            &d_slow,
            len,
            first_valid,
            combos.len(),
            &mut d_fast_out,
            &mut d_slow_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;

        Ok((
            DeviceArrayF32 {
                buf: d_fast_out,
                rows: combos.len(),
                cols: len,
            },
            DeviceArrayF32 {
                buf: d_slow_out,
                rows: combos.len(),
                cols: len,
            },
        ))
    }

    pub fn buff_averages_batch_dev(
        &self,
        price_f32: &[f32],
        volume_f32: &[f32],
        sweep: &BuffAveragesBatchRange,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32), CudaBuffAveragesError> {
        let (combos, first_valid, _len) = Self::prepare_batch_inputs(price_f32, volume_f32, sweep)?;
        self.run_batch_kernel(price_f32, volume_f32, &combos, first_valid)
    }

    pub fn buff_averages_batch_into_host_f32(
        &self,
        price_f32: &[f32],
        volume_f32: &[f32],
        sweep: &BuffAveragesBatchRange,
        fast_out: &mut [f32],
        slow_out: &mut [f32],
    ) -> Result<(usize, usize, Vec<(usize, usize)>), CudaBuffAveragesError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(price_f32, volume_f32, sweep)?;
        let expected = combos.len() * len;
        if fast_out.len() != expected || slow_out.len() != expected {
            return Err(CudaBuffAveragesError::InvalidInput(format!(
                "output slice mismatch (expected {}, fast={}, slow={})",
                expected,
                fast_out.len(),
                slow_out.len()
            )));
        }

        let (fast_dev, slow_dev) =
            self.run_batch_kernel(price_f32, volume_f32, &combos, first_valid)?;
        fast_dev
            .buf
            .copy_to(fast_out)
            .map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        slow_dev
            .buf
            .copy_to(slow_out)
            .map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;

        Ok((combos.len(), len, combos))
    }
}
