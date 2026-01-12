use vector_ta::indicators::dec_osc::{dec_osc, DecOscBatchBuilder, DecOscInput, DecOscParams};

#[test]
fn test_dec_osc_verification() {
    let data = vec![
        2761.7, 2740.0, 2763.0, 2800.0, 2750.0, 2780.0, 2820.0, 2810.0, 2795.0, 2830.0, 2850.0,
        2840.0, 2860.0, 2880.0, 2875.0,
    ];

    let params = DecOscParams {
        hp_period: Some(10),
        k: Some(1.0),
    };
    let input = DecOscInput::from_slice(&data, params);
    let result = dec_osc(&input).unwrap();

    assert!(result.values[0].is_nan(), "Expected NaN at index 0");
    assert!(result.values[1].is_nan(), "Expected NaN at index 1");
    assert!(!result.values[2].is_nan(), "Expected value at index 2");

    let batch_result = DecOscBatchBuilder::new()
        .hp_period_range(5, 10, 5)
        .k_range(0.5, 1.5, 0.5)
        .apply_slice(&data)
        .unwrap();

    let expected_combos = 2 * 3;
    assert_eq!(batch_result.combos.len(), expected_combos);
    assert_eq!(batch_result.values.len(), expected_combos * data.len());

    for (idx, combo) in batch_result.combos.iter().enumerate() {
        let single_params = DecOscParams {
            hp_period: combo.hp_period,
            k: combo.k,
        };
        let single_input = DecOscInput::from_slice(&data, single_params);
        let single_result = dec_osc(&single_input).unwrap();

        let row_start = idx * data.len();
        let row_end = row_start + data.len();
        let batch_row = &batch_result.values[row_start..row_end];

        for i in 0..data.len() {
            if single_result.values[i].is_nan() && batch_row[i].is_nan() {
                continue;
            }
            let diff = (single_result.values[i] - batch_row[i]).abs();
            assert!(
                diff < 1e-10,
                "Mismatch at combo {} pos {}: single={}, batch={}",
                idx,
                i,
                single_result.values[i],
                batch_row[i]
            );
        }
    }
}

#[test]
fn test_dec_osc_api_completeness() {
    use vector_ta::indicators::dec_osc::{
        dec_osc_into_slice, dec_osc_with_kernel, DecOscBuilder, DecOscStream,
    };
    use vector_ta::utilities::enums::Kernel;

    let data = vec![100.0; 50];

    let builder_result = DecOscBuilder::new()
        .hp_period(10)
        .k(1.0)
        .kernel(Kernel::Scalar)
        .apply_slice(&data)
        .unwrap();
    assert_eq!(builder_result.values.len(), data.len());

    let stream_params = DecOscParams {
        hp_period: Some(10),
        k: Some(1.0),
    };
    let mut stream = DecOscStream::try_new(stream_params).unwrap();
    let mut stream_results = Vec::new();
    for &val in &data {
        stream_results.push(stream.update(val));
    }
    assert_eq!(stream_results.len(), data.len());

    let params = DecOscParams {
        hp_period: Some(10),
        k: Some(1.0),
    };
    let input = DecOscInput::from_slice(&data, params);
    let kernel_result = dec_osc_with_kernel(&input, Kernel::Scalar).unwrap();
    assert_eq!(kernel_result.values.len(), data.len());

    let mut output = vec![0.0; data.len()];
    dec_osc_into_slice(&mut output, &input, Kernel::Scalar).unwrap();
    assert_eq!(output.len(), data.len());
}

#[test]
fn test_dec_osc_error_handling() {
    use vector_ta::indicators::dec_osc::DecOscError;

    let all_nan = vec![f64::NAN; 10];
    let params = DecOscParams::default();
    let input = DecOscInput::from_slice(&all_nan, params);
    let result = dec_osc(&input);
    assert!(matches!(result, Err(DecOscError::AllValuesNaN)));

    let data = vec![100.0; 5];
    let params = DecOscParams {
        hp_period: Some(0),
        k: Some(1.0),
    };
    let input = DecOscInput::from_slice(&data, params);
    let result = dec_osc(&input);
    assert!(matches!(result, Err(DecOscError::InvalidPeriod { .. })));

    let long_data = vec![100.0; 20];
    let params = DecOscParams {
        hp_period: Some(10),
        k: Some(0.0),
    };
    let input = DecOscInput::from_slice(&long_data, params);
    let result = dec_osc(&input);
    assert!(
        matches!(result, Err(DecOscError::InvalidK { k }) if k == 0.0),
        "Expected InvalidK error for k=0.0, got: {:?}",
        result
    );

    let small_data = vec![100.0];
    let params = DecOscParams {
        hp_period: Some(10),
        k: Some(1.0),
    };
    let input = DecOscInput::from_slice(&small_data, params);
    let result = dec_osc(&input);

    assert!(
        matches!(result, Err(DecOscError::InvalidPeriod { .. })),
        "Expected InvalidPeriod error for period > data_len, got: {:?}",
        result
    );
}
