use my_project::other_indicators::fvg_trailing_stop::{
    fvg_trailing_stop, FvgTrailingStopInput, FvgTrailingStopParams,
};
use my_project::utilities::data_loader::read_candles_from_csv;

fn main() {
    let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
    let candles = read_candles_from_csv(file_path).unwrap();
    
    let params = FvgTrailingStopParams {
        unmitigated_fvg_lookback: Some(5),
        smoothing_length: Some(9),
        reset_on_cross: Some(false),
    };
    let smoothing_len = params.smoothing_length.unwrap_or(9);
    let input = FvgTrailingStopInput::from_candles(&candles, params);
    let result = fvg_trailing_stop(&input).unwrap();
    
    let n = result.lower.len();
    println!("Total data points: {}", n);
    println!("\nLast 5 values:");
    
    for i in (n-5)..n {
        println!("Index {}: upper={:.8}, lower={:.8}, upper_ts={:.8}, lower_ts={:.8}", 
                 i, 
                 if result.upper[i].is_nan() { f64::NAN } else { result.upper[i] },
                 if result.lower[i].is_nan() { f64::NAN } else { result.lower[i] },
                 if result.upper_ts[i].is_nan() { f64::NAN } else { result.upper_ts[i] },
                 if result.lower_ts[i].is_nan() { f64::NAN } else { result.lower_ts[i] });
    }
    
    // Also check warmup period
    let warmup = 2 + smoothing_len - 1;
    println!("\nExpected warmup period: {} values", warmup);
    
    let mut first_non_nan = None;
    for i in 0..n {
        if !result.upper[i].is_nan() || !result.lower[i].is_nan() || 
           !result.upper_ts[i].is_nan() || !result.lower_ts[i].is_nan() {
            first_non_nan = Some(i);
            break;
        }
    }
    
    println!("First non-NaN value at index: {:?}", first_non_nan);
}