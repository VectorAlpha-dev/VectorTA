use vector_ta::indicators::leavitt_convolution_acceleration::{
    leavitt_convolution_acceleration, LeavittConvolutionAccelerationBatchRange,
    LeavittConvolutionAccelerationInput, LeavittConvolutionAccelerationParams,
};

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaLeavittConvolutionAcceleration};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        (a - b).abs() <= tol
    }
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    assert!(true);
}

#[cfg(feature = "cuda")]
#[test]
fn leavitt_convolution_acceleration_cuda_batch_matches_cpu(
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!(
            "[leavitt_convolution_acceleration_cuda_batch_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let len = 768usize;
    let mut data = vec![f64::NAN; len];
    let mut base = 82.0f64;
    for i in 20..len {
        let x = i as f64;
        base += (x * 0.009).sin() * 0.31 + (x * 0.002).cos() * 0.08;
        data[i] = base + (x * 0.017).sin() * 0.82 + (x * 0.004).cos() * 0.27;
    }

    let cuda =
        CudaLeavittConvolutionAcceleration::new(0).expect("CudaLeavittConvolutionAcceleration");

    for use_norm_hyperbolic in [true, false] {
        let sweep = LeavittConvolutionAccelerationBatchRange {
            length: (32, 36, 4),
            norm_length: (45, 55, 10),
            use_norm_hyperbolic: Some(use_norm_hyperbolic),
        };

        let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

        assert_eq!(result.outputs.rows(), result.combos.len());
        assert_eq!(result.outputs.cols(), data.len());

        let mut got_conv = vec![0.0f64; result.outputs.conv_acceleration.len()];
        let mut got_signal = vec![0.0f64; result.outputs.signal.len()];
        result
            .outputs
            .conv_acceleration
            .buf
            .copy_to(&mut got_conv)?;
        result.outputs.signal.buf.copy_to(&mut got_signal)?;

        for (row, params) in result.combos.iter().enumerate() {
            let cpu = leavitt_convolution_acceleration(
                &LeavittConvolutionAccelerationInput::from_slice(
                    &data,
                    LeavittConvolutionAccelerationParams {
                        length: params.length,
                        norm_length: params.norm_length,
                        use_norm_hyperbolic: params.use_norm_hyperbolic,
                    },
                ),
            )?;
            let start = row * data.len();

            for idx in 0..data.len() {
                assert!(
                    approx_eq(cpu.conv_acceleration[idx], got_conv[start + idx], 1e-8),
                    "conv_acceleration mismatch for use_norm_hyperbolic={use_norm_hyperbolic}, row={row}, idx={idx}: cpu={} cuda={}",
                    cpu.conv_acceleration[idx],
                    got_conv[start + idx]
                );
                assert!(
                    approx_eq(cpu.signal[idx], got_signal[start + idx], 1e-8),
                    "signal mismatch for use_norm_hyperbolic={use_norm_hyperbolic}, row={row}, idx={idx}: cpu={} cuda={}",
                    cpu.signal[idx],
                    got_signal[start + idx]
                );
            }
        }
    }

    Ok(())
}
