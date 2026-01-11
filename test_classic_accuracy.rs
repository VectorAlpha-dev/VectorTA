
use my_project::indicators::zscore::{zscore, ZscoreInput, ZscoreParams};
use my_project::indicators::bollinger_bands::{bollinger_bands, BollingerBandsInput, BollingerBandsParams};

fn main() {
    let data = vec![
        10.0, 12.0, 11.5, 13.0, 14.0, 13.5, 12.5, 11.0, 12.0, 13.0,
        14.5, 15.0, 14.0, 13.5, 12.0, 11.5, 13.0, 14.0, 15.5, 16.0
    ];

    
    let zscore_input = ZscoreInput::from_slice(&data, ZscoreParams {
        period: Some(5),
        ma_type: Some("sma".to_string()),
        nbdev: Some(1.0),
        devtype: Some(0),
    });

    let zscore_result = zscore(&zscore_input).unwrap();
    println!("Zscore SMA results:");
    for (i, val) in zscore_result.values.iter().enumerate() {
        if !val.is_nan() {
            println!("  [{}]: {:.6}", i, val);
        }
    }

    
    let bb_input = BollingerBandsInput::from_slice(&data, BollingerBandsParams {
        period: Some(5),
        devup: Some(2.0),
        devdn: Some(2.0),
        matype: Some("sma".to_string()),
        devtype: Some(0),
    });

    let bb_result = bollinger_bands(&bb_input).unwrap();
    println!("\nBollinger Bands SMA results:");
    for i in 0..data.len() {
        if !bb_result.middle[i].is_nan() {
            println!("  [{}]: upper={:.6}, middle={:.6}, lower={:.6}",
                i, bb_result.upper[i], bb_result.middle[i], bb_result.lower[i]);
        }
    }
}
