use my_project::indicators::moving_averages::hma::{hma_with_kernel, HmaInput, HmaParams};
use my_project::utilities::data_loader::read_candles_from_csv;
use my_project::utilities::enums::Kernel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
    let candles = read_candles_from_csv(file_path)?;
    
    // Test multiple periods
    for period in [2, 3, 5, 10, 20, 50] {
        let data_slice = &candles.close[0..500];
        let params = HmaParams { period: Some(period) };
        let input = HmaInput::from_slice(data_slice, params);
        
        let scalar_result = hma_with_kernel(&input, Kernel::Scalar)?;
        let avx512_result = hma_with_kernel(&input, Kernel::Avx512)?;
        
        let mut max_ulp = 0u64;
        let mut max_diff = 0.0;
        let mut problem_count = 0;
        
        for i in 0..scalar_result.values.len() {
            let s = scalar_result.values[i];
            let a5 = avx512_result.values[i];
            
            if s.is_nan() {
                continue;
            }
            
            let ulp = s.to_bits().abs_diff(a5.to_bits());
            let diff = (s - a5).abs();
            
            if ulp > 8 {
                problem_count += 1;
            }
            
            if ulp > max_ulp {
                max_ulp = ulp;
                max_diff = diff;
            }
        }
        
        println!("Period {:3}: max ULP={:6}, max diff={:.15}, problems={}", 
                 period, max_ulp, max_diff, problem_count);
    }
    
    Ok(())
}