// Helper script to generate the complete measure_and_collect function
// This generates the code needed to measure all indicators

use std::collections::HashSet;

fn main() {
    let indicators = vec![
        ("sma", "SmaInput::from_slice(&data.close, SmaParams { period: Some(14) })", "sma::sma"),
        ("ema", "EmaInput::from_slice(&data.close, EmaParams { period: Some(14) })", "ema::ema"),
        ("rsi", "RsiInput::from_slice(&data.close, RsiParams { period: Some(14) })", "rsi::rsi"),
        ("bollinger_bands", "BollingerBandsInput::from_slice(&data.close, BollingerBandsParams { period: Some(20), devup: Some(2.0), devdn: Some(2.0), matype: Some(\"sma\".to_string()), devtype: Some(0) })", "bollinger_bands::bollinger_bands"),
        ("macd", "MacdInput::from_slice(&data.close, MacdParams { fast_period: Some(12), slow_period: Some(26), signal_period: Some(9), ma_type: None })", "macd::macd"),
        ("atr", "AtrInput::from_candles(&candles, AtrParams { length: Some(14) })", "atr::atr"),
        ("stoch", "StochInput::from_candles(&candles, StochParams { fastk_period: Some(14), slowk_period: Some(3), slowk_matype: None, slowd_period: Some(3), slowd_matype: None })", "stoch::stoch"),
        ("aroon", "AroonInput::from_candles(&candles, AroonParams { length: Some(14) })", "aroon::aroon"),
        ("adx", "AdxInput::from_candles(&candles, AdxParams { period: Some(14) })", "adx::adx"),
        ("cci", "CciInput::from_candles(&candles, \"hlc3\", CciParams { period: Some(20) })", "cci::cci"),
        ("dema", "DemaInput::from_slice(&data.close, DemaParams { period: Some(14) })", "dema::dema"),
        ("tema", "TemaInput::from_slice(&data.close, TemaParams { period: Some(14) })", "tema::tema"),
        ("wma", "WmaInput::from_slice(&data.close, WmaParams { period: Some(14) })", "wma::wma"),
        ("kama", "KamaInput::from_slice(&data.close, KamaParams { period: Some(14) })", "kama::kama"),
        ("trima", "TrimaInput::from_slice(&data.close, TrimaParams { period: Some(14) })", "trima::trima"),
        ("hma", "HmaInput::from_slice(&data.close, HmaParams { period: Some(14) })", "hma::hma"),
        ("zlema", "ZlemaInput::from_slice(&data.close, ZlemaParams { period: Some(14) })", "zlema::zlema"),
        ("vwma", "VwmaInput::from_candles(&candles, \"close\", VwmaParams { period: Some(14) })", "vwma::vwma"),
        ("wilders", "WildersInput::from_slice(&data.close, WildersParams { period: Some(14) })", "wilders::wilders"),
        ("apo", "ApoInput::from_slice(&data.close, ApoParams { fast: Some(12), slow: Some(26), matype: None })", "apo::apo"),
        ("cmo", "CmoInput::from_slice(&data.close, CmoParams { period: Some(14) })", "cmo::cmo"),
        ("dpo", "DpoInput::from_slice(&data.close, DpoParams { period: Some(14) })", "dpo::dpo"),
        ("mom", "MomInput::from_slice(&data.close, MomParams { period: Some(10) })", "mom::mom"),
        ("ppo", "PpoInput::from_slice(&data.close, PpoParams { fast_period: Some(12), slow_period: Some(26), ma_type: None })", "ppo::ppo"),
        ("roc", "RocInput::from_slice(&data.close, RocParams { period: Some(10) })", "roc::roc"),
        ("rocr", "RocrInput::from_slice(&data.close, RocrParams { period: Some(10) })", "rocr::rocr"),
        ("rocp", "RocpInput::from_slice(&data.close, RocpParams { period: Some(10) })", "rocp::rocp"),
        ("willr", "WillrInput::from_candles(&candles, WillrParams { period: Some(14) })", "willr::willr"),
        ("ad", "AdInput::from_candles(&candles, AdParams {})", "ad::ad"),
        ("adosc", "AdoscInput::from_candles(&candles, AdoscParams { fast_period: Some(3), slow_period: Some(10) })", "adosc::adosc"),
        ("obv", "ObvInput::from_candles(&candles, ObvParams {})", "obv::obv"),
        ("mfi", "MfiInput::from_candles(&candles, \"hlc3\", MfiParams { period: Some(14) })", "mfi::mfi"),
        ("ao", "AoInput::from_candles(&candles, \"hl2\", AoParams { short_period: Some(5), long_period: Some(34) })", "ao::ao"),
        ("bop", "BopInput::from_candles(&candles, BopParams {})", "bop::bop"),
        ("natr", "NatrInput::from_candles(&candles, NatrParams { period: Some(14) })", "natr::natr"),
        ("stddev", "StddevInput::from_slice(&data.close, StddevParams { period: Some(5) })", "stddev::stddev"),
        ("var", "VarInput::from_slice(&data.close, VarParams { period: Some(5) })", "var::var"),
        ("ultosc", "UltoscInput::from_candles(&candles, UltoscParams { short_period: Some(7), medium_period: Some(14), long_period: Some(28) })", "ultosc::ultosc"),
        ("adxr", "AdxrInput::from_candles(&candles, AdxrParams { period: Some(14) })", "adxr::adxr"),
        ("aroonosc", "AroonoscInput::from_candles(&candles, AroonoscParams { period: Some(14) })", "aroonosc::aroonosc"),
        ("di", "DiInput::from_candles(&candles, DiParams { period: Some(14) })", "di::di"),
        ("dm", "DmInput::from_candles(&candles, DmParams { period: Some(14) })", "dm::dm"),
        ("dx", "DxInput::from_candles(&candles, DxParams { period: Some(14) })", "dx::dx"),
        ("fisher", "FisherInput::from_candles(&candles, \"hl2\", FisherParams { period: Some(14) })", "fisher::fisher"),
        ("fosc", "FoscInput::from_slice(&data.close, FoscParams { period: Some(14) })", "fosc::fosc"),
        ("kvo", "KvoInput::from_candles(&candles, KvoParams { short_period: Some(34), long_period: Some(55), signal_period: Some(13) })", "kvo::kvo"),
        ("linearreg_slope", "LinearregSlopeInput::from_slice(&data.close, LinearregSlopeParams { period: Some(14) })", "linearreg_slope::linearreg_slope"),
        ("linearreg_intercept", "LinearregInterceptInput::from_slice(&data.close, LinearregInterceptParams { period: Some(14) })", "linearreg_intercept::linearreg_intercept"),
        ("mass", "MassInput::from_candles(&candles, \"hl2\", MassParams { high_period: Some(9), low_period: Some(25) })", "mass::mass"),
        ("medprice", "MedpriceInput::from_candles(&candles, MedpriceParams {})", "medprice::medprice"),
        ("midpoint", "MidpointInput::from_slice(&data.close, MidpointParams { period: Some(14) })", "midpoint::midpoint"),
        ("midprice", "MidpriceInput::from_candles(&candles, MidpriceParams { period: Some(14) })", "midprice::midprice"),
        ("nvi", "NviInput::from_candles(&candles, NviParams {})", "nvi::nvi"),
        ("pvi", "PviInput::from_candles(&candles, PviParams {})", "pvi::pvi"),
        ("qstick", "QstickInput::from_candles(&candles, QstickParams { period: Some(14) })", "qstick::qstick"),
        ("sar", "SarInput::from_candles(&candles, SarParams { acceleration: Some(0.02), maximum: Some(0.2) })", "sar::sar"),
        ("srsi", "SrsiInput::from_slice(&data.close, SrsiParams { rsi_period: Some(14), stoch_period: Some(14), k_period: Some(3), d_period: Some(3) })", "srsi::srsi"),
        ("stochf", "StochfInput::from_candles(&candles, StochfParams { fastk_period: Some(5), fastd_period: Some(3), fastd_matype: None })", "stochf::stochf"),
        ("trix", "TrixInput::from_slice(&data.close, TrixParams { period: Some(14) })", "trix::trix"),
        ("tsf", "TsfInput::from_slice(&data.close, TsfParams { period: Some(14) })", "tsf::tsf"),
        ("vidya", "VidyaInput::from_slice(&data.close, VidyaParams { short_period: Some(2), long_period: Some(5), alpha: Some(0.2) })", "vidya::vidya"),
        ("vosc", "VoscInput::from_candles(&candles, VoscParams { short_period: Some(5), long_period: Some(10) })", "vosc::vosc"),
        ("wad", "WadInput::from_candles(&candles, WadParams {})", "wad::wad"),
        ("wclprice", "WclpriceInput::from_candles(&candles, WclpriceParams {})", "wclprice::wclprice"),
        ("cvi", "CviInput::from_candles(&candles, CviParams { period: Some(14) })", "cvi::cvi"),
        ("emv", "EmvInput::from_candles(&candles, EmvParams {})", "emv::emv"),
        ("marketefi", "MarketefiInput::from_candles(&candles, MarketefiParams {})", "marketefi::marketefi"),
        ("minmax", "MinmaxInput::from_slice(&data.close, MinmaxParams { period: Some(14) })", "minmax::minmax"),
        ("msw", "MswInput::from_slice(&data.close, MswParams { period: Some(14) })", "msw::msw"),
    ];

    println!("// Generated measure_and_collect function");
    println!("fn measure_and_collect(indicator: &IndicatorMapping, data: &CandleData, _size_name: &str) {{");
    println!("    let iterations = 10;");
    println!("    let candles = create_candles(data);");
    println!("    ");
    println!("    // Measure Rust Native");
    println!("    match indicator.rust_name {{");
    
    for (name, input_code, func) in &indicators {
        println!("        \"{}\" => {{", name);
        println!("            let input = {};", input_code);
        println!("            let start = Instant::now();");
        println!("            for _ in 0..iterations {{");
        println!("                let _ = black_box({}(&input));", func);
        println!("            }}");
        println!("            let duration = start.elapsed() / iterations as u32;");
        println!("            COLLECTOR.add_measurement(indicator.rust_name, LibraryType::RustNative, duration, data.len());");
        println!("        }}");
    }
    
    println!("        _ => {{}}  // Skip unmapped indicators");
    println!("    }}");
    println!("    ");
    println!("    // Measure Tulip (existing code remains the same)");
    println!("    unsafe {{");
    println!("        // ... (keep existing Tulip measurement code)");
    println!("    }}");
    println!("}}");
}