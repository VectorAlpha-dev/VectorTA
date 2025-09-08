use my_project::other_indicators::fvg_trailing_stop::{FvgTrailingStopInput, FvgTrailingStopParams, fvg_trailing_stop_with_kernel};
use my_project::utilities::data_loader::read_candles_from_csv;
use my_project::utilities::enums::Kernel;

fn main() {
    let candles = read_candles_from_csv("src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv").unwrap();
    let params = FvgTrailingStopParams {
        unmitigated_fvg_lookback: Some(5),
        smoothing_length: Some(9),
        reset_on_cross: Some(false),
    };
    let input = FvgTrailingStopInput::from_candles(&candles, params);
    let result = fvg_trailing_stop_with_kernel(&input, Kernel::Scalar).unwrap();
    
    let n = result.lower.len();
    println!("Total data points: {}", n);
    println!("\nLast 5 values:");
    for i in (n-5)..n {
        println!("Index {}: lower={:.8}, lower_ts={:.8}", 
                i, 
                if result.lower[i].is_nan() { "NaN".to_string() } else { format!("{:.8}", result.lower[i]) },
                if result.lower_ts[i].is_nan() { "NaN".to_string() } else { format!("{:.8}", result.lower_ts[i]) });
    }
}
