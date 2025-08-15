use my_project::indicators::moving_averages::hma::{hma_with_kernel, HmaInput, HmaParams};
use my_project::utilities::data_loader::read_candles_from_csv;
use my_project::utilities::enums::Kernel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load real market data
    let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
    let candles = read_candles_from_csv(file_path)?;
    
    // Test with period=2 (the failing case from proptest)
    let period = 2;
    let start_idx = 0;
    let end_idx = 200;
    
    let data_slice = &candles.close[start_idx..end_idx];
    let params = HmaParams { period: Some(period) };
    let input = HmaInput::from_slice(data_slice, params);
    
    // Compute with different kernels
    let scalar_result = hma_with_kernel(&input, Kernel::Scalar)?;
    let avx2_result = hma_with_kernel(&input, Kernel::Avx2)?;
    let avx512_result = hma_with_kernel(&input, Kernel::Avx512)?;
    
    // Find discrepancies
    let mut max_diff_avx2 = 0.0;
    let mut max_diff_avx512 = 0.0;
    let mut max_ulp_avx2 = 0u64;
    let mut max_ulp_avx512 = 0u64;
    let mut problem_idx = None;
    
    for i in 0..scalar_result.values.len() {
        let s = scalar_result.values[i];
        let a2 = avx2_result.values[i];
        let a5 = avx512_result.values[i];
        
        if s.is_nan() {
            continue;
        }
        
        let diff_avx2 = (s - a2).abs();
        let diff_avx512 = (s - a5).abs();
        
        let ulp_avx2 = s.to_bits().abs_diff(a2.to_bits());
        let ulp_avx512 = s.to_bits().abs_diff(a5.to_bits());
        
        if diff_avx2 > max_diff_avx2 {
            max_diff_avx2 = diff_avx2;
            max_ulp_avx2 = ulp_avx2;
        }
        
        if diff_avx512 > max_diff_avx512 {
            max_diff_avx512 = diff_avx512;
            max_ulp_avx512 = ulp_avx512;
            if ulp_avx512 > 8 {
                problem_idx = Some(i);
            }
        }
    }
    
    println!("HMA AVX512 Debugging for period={}", period);
    println!("Data length: {}", data_slice.len());
    println!("\nMax differences from Scalar:");
    println!("  AVX2:   diff={:.15}, ULP={}", max_diff_avx2, max_ulp_avx2);
    println!("  AVX512: diff={:.15}, ULP={}", max_diff_avx512, max_ulp_avx512);
    
    if let Some(idx) = problem_idx {
        println!("\nProblem at index {}:", idx);
        println!("  Input window: {:?}", &data_slice[idx.saturating_sub(5)..=idx.min(data_slice.len()-1)]);
        println!("  Scalar: {:.15}", scalar_result.values[idx]);
        println!("  AVX2:   {:.15}", avx2_result.values[idx]);
        println!("  AVX512: {:.15}", avx512_result.values[idx]);
        
        // Check warmup calculation
        let sqrt_period = (period as f64).sqrt().floor() as usize;
        let expected_warmup = period + sqrt_period - 1;
        println!("\nExpected warmup: {}", expected_warmup);
        println!("First non-NaN:");
        for (name, vals) in [("Scalar", &scalar_result.values), ("AVX2", &avx2_result.values), ("AVX512", &avx512_result.values)] {
            if let Some(first) = vals.iter().position(|x| !x.is_nan()) {
                println!("  {}: index {}", name, first);
            }
        }
    }
    
    Ok(())
}