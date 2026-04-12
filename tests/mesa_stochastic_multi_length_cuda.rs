use vector_ta::indicators::mesa_stochastic_multi_length::{
    mesa_stochastic_multi_length_batch_with_kernel, MesaStochasticMultiLengthBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaMesaStochasticMultiLength};

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
fn mesa_stochastic_multi_length_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[mesa_stochastic_multi_length_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 1536usize;
    let mut source = vec![0.0f64; len];
    let mut base = 96.0f64;
    for i in 0..len {
        let x = i as f64;
        base += (x * 0.009).sin() * 0.34 + (x * 0.0027).cos() * 0.12;
        source[i] = base + (x * 0.021).sin() * 0.91 + (x * 0.004).cos() * 0.28;
    }
    source[320] = f64::NAN;
    source[321] = f64::NAN;
    source[997] = f64::NAN;

    let sweep = MesaStochasticMultiLengthBatchRange {
        length_1: (48, 50, 2),
        length_2: (21, 21, 0),
        length_3: (9, 9, 0),
        length_4: (6, 6, 0),
        trigger_length: (2, 3, 1),
    };

    let cpu = mesa_stochastic_multi_length_batch_with_kernel(&source, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaMesaStochasticMultiLength::new(0).expect("CudaMesaStochasticMultiLength::new");
    let result = cuda.batch_dev(&source, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_mesa_1 = vec![0.0f64; result.outputs.mesa_1.len()];
    let mut got_mesa_2 = vec![0.0f64; result.outputs.mesa_2.len()];
    let mut got_mesa_3 = vec![0.0f64; result.outputs.mesa_3.len()];
    let mut got_mesa_4 = vec![0.0f64; result.outputs.mesa_4.len()];
    let mut got_trigger_1 = vec![0.0f64; result.outputs.trigger_1.len()];
    let mut got_trigger_2 = vec![0.0f64; result.outputs.trigger_2.len()];
    let mut got_trigger_3 = vec![0.0f64; result.outputs.trigger_3.len()];
    let mut got_trigger_4 = vec![0.0f64; result.outputs.trigger_4.len()];
    result.outputs.mesa_1.buf.copy_to(&mut got_mesa_1)?;
    result.outputs.mesa_2.buf.copy_to(&mut got_mesa_2)?;
    result.outputs.mesa_3.buf.copy_to(&mut got_mesa_3)?;
    result.outputs.mesa_4.buf.copy_to(&mut got_mesa_4)?;
    result.outputs.trigger_1.buf.copy_to(&mut got_trigger_1)?;
    result.outputs.trigger_2.buf.copy_to(&mut got_trigger_2)?;
    result.outputs.trigger_3.buf.copy_to(&mut got_trigger_3)?;
    result.outputs.trigger_4.buf.copy_to(&mut got_trigger_4)?;

    for idx in 0..cpu.mesa_1.len() {
        assert!(
            approx_eq(cpu.mesa_1[idx], got_mesa_1[idx], 1e-10),
            "mesa_1 mismatch at {idx}: cpu={} cuda={}",
            cpu.mesa_1[idx],
            got_mesa_1[idx]
        );
        assert!(
            approx_eq(cpu.mesa_2[idx], got_mesa_2[idx], 1e-10),
            "mesa_2 mismatch at {idx}: cpu={} cuda={}",
            cpu.mesa_2[idx],
            got_mesa_2[idx]
        );
        assert!(
            approx_eq(cpu.mesa_3[idx], got_mesa_3[idx], 1e-10),
            "mesa_3 mismatch at {idx}: cpu={} cuda={}",
            cpu.mesa_3[idx],
            got_mesa_3[idx]
        );
        assert!(
            approx_eq(cpu.mesa_4[idx], got_mesa_4[idx], 1e-10),
            "mesa_4 mismatch at {idx}: cpu={} cuda={}",
            cpu.mesa_4[idx],
            got_mesa_4[idx]
        );
        assert!(
            approx_eq(cpu.trigger_1[idx], got_trigger_1[idx], 1e-10),
            "trigger_1 mismatch at {idx}: cpu={} cuda={}",
            cpu.trigger_1[idx],
            got_trigger_1[idx]
        );
        assert!(
            approx_eq(cpu.trigger_2[idx], got_trigger_2[idx], 1e-10),
            "trigger_2 mismatch at {idx}: cpu={} cuda={}",
            cpu.trigger_2[idx],
            got_trigger_2[idx]
        );
        assert!(
            approx_eq(cpu.trigger_3[idx], got_trigger_3[idx], 1e-10),
            "trigger_3 mismatch at {idx}: cpu={} cuda={}",
            cpu.trigger_3[idx],
            got_trigger_3[idx]
        );
        assert!(
            approx_eq(cpu.trigger_4[idx], got_trigger_4[idx], 1e-10),
            "trigger_4 mismatch at {idx}: cpu={} cuda={}",
            cpu.trigger_4[idx],
            got_trigger_4[idx]
        );
    }

    Ok(())
}
