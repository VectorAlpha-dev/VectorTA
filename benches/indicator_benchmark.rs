extern crate criterion;
extern crate lazy_static;
extern crate my_project;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use my_project::indicators::data_loader::read_candles_from_csv;

use my_project::indicators::{
    acosc::{calculate_acosc, AcoscInput},
    ad::{calculate_ad, AdInput},
    adosc::{calculate_adosc, AdoscInput},
    adx::{calculate_adx, AdxInput},
    adxr::{calculate_adxr, AdxrInput},
    alligator::{calculate_alligator, AlligatorInput},
    alma::{calculate_alma, AlmaInput},
    ao::{calculate_ao, AoInput},
    apo::{calculate_apo, ApoInput},
    aroon::{calculate_aroon, AroonInput},
    aroonosc::{calculate_aroon_osc, AroonOscInput},
    atr::{calculate_atr, AtrInput},
    avgprice::{calculate_avgprice, AvgPriceInput},
    bandpass::{calculate_bandpass, BandPassInput},
    dema::{calculate_dema, DemaInput},
    ema::{calculate_ema, EmaInput},
    fwma::{calculate_fwma, FwmaInput},
    gaussian::{calculate_gaussian, GaussianInput},
    highpass::{calculate_highpass, HighPassInput},
    highpass_2_pole::{calculate_high_pass_2_pole, HighPass2Input},
    hma::{calculate_hma, HmaInput},
    ht_trendline::{calculate_ht_trendline, EhlersITrendInput},
    jma::{calculate_jma, JmaInput},
    kama::{calculate_kama, KamaInput},
    linearreg::{calculate_linreg, LinRegInput},
    mama::{calculate_mama, MamaInput},
    pwma::{calculate_pwma, PwmaInput},
    reflex::{calculate_reflex, ReflexInput},
    rsi::{calculate_rsi, RsiInput},
    sinwma::{calculate_sinwma, SinWmaInput},
    sma::{calculate_sma, SmaInput},
    smma::{calculate_smma, SmmaInput},
    supersmoother::{calculate_supersmoother, SuperSmootherInput},
    supersmoother_3_pole::{calculate_supersmoother_3_pole, SuperSmoother3PoleInput},
    tema::{calculate_tema, TemaInput},
    tilson::{calculate_t3, T3Input},
    trendflex::{calculate_trendflex, TrendFlexInput},
    trima::{calculate_trima, TrimaInput},
    vwma::{calculate_vwma, VwmaInput},
    wilders::{calculate_wilders, WildersInput},
    wma::{calculate_wma, WmaInput},
    zlema::{calculate_zlema, ZlemaInput},
};
use std::time::Duration;

fn benchmark_indicators(c: &mut Criterion) {
    let candles =
        read_candles_from_csv("src/data/bitfinex btc-usd 100,000 candles ends 09-01-24.csv")
            .expect("Failed to load candles");

    let close_prices = candles
        .select_candle_field("close")
        .expect("Failed to extract close prices");
    let hl2_prices = candles
        .get_calculated_field("hl2")
        .expect("Failed to extract hl2 prices");

    let mut group = c.benchmark_group("Indicator Benchmarks");
    group.measurement_time(Duration::new(8, 0));
    group.warm_up_time(Duration::new(4, 0));

    // TrendFlex
    group.bench_function(BenchmarkId::new("TRENDFLEX", 0), |b| {
        let input = TrendFlexInput::with_default_params(close_prices);
        b.iter(|| calculate_trendflex(black_box(&input)).expect("Failed to calculate TRENDFLEX"))
    });

    // VWMA
    group.bench_function(BenchmarkId::new("VWMA", 0), |b| {
        let input = VwmaInput::with_default_params(&candles);
        b.iter(|| calculate_vwma(black_box(&input)).expect("Failed to calculate VWMA"))
    });

    // PWMA
    group.bench_function(BenchmarkId::new("PWMA", 0), |b| {
        let input = PwmaInput::with_default_params(close_prices);
        b.iter(|| calculate_pwma(black_box(&input)).expect("Failed to calculate PWMA"))
    });

    // HT Trendline
    group.bench_function(BenchmarkId::new("HT_TRENDLINE", 0), |b| {
        let input = EhlersITrendInput::with_default_params(close_prices);
        b.iter(|| {
            calculate_ht_trendline(black_box(&input)).expect("Failed to calculate HT_TRENDLINE")
        })
    });

    // SMMA
    group.bench_function(BenchmarkId::new("SMMA", 0), |b| {
        let input = SmmaInput::with_default_params(close_prices);
        b.iter(|| calculate_smma(black_box(&input)).expect("Failed to calculate SMMA"))
    });

    // Reflex
    group.bench_function(BenchmarkId::new("REFLEX", 0), |b| {
        let input = ReflexInput::with_default_params(close_prices);
        b.iter(|| calculate_reflex(black_box(&input)).expect("Failed to calculate REFLEX"))
    });

    // JMA
    group.bench_function(BenchmarkId::new("JMA", 0), |b| {
        let input = JmaInput::with_default_params(close_prices);
        b.iter(|| calculate_jma(black_box(&input)).expect("Failed to calculate JMA"))
    });

    // High Pass 2 Pole
    group.bench_function(BenchmarkId::new("HIGHPASS_2Pole", 0), |b| {
        let input = HighPass2Input::with_default_params(close_prices);
        b.iter(|| {
            calculate_high_pass_2_pole(black_box(&input)).expect("Failed to calculate HIGHPASS2")
        })
    });

    // High Pass
    group.bench_function(BenchmarkId::new("HIGHPASS_1Pole", 0), |b| {
        let input = HighPassInput::with_default_params(close_prices);
        b.iter(|| calculate_highpass(black_box(&input)).expect("Failed to calculate HIGHPASS"))
    });

    // Gaussian
    group.bench_function(BenchmarkId::new("GAUSSIAN", 0), |b| {
        let input = GaussianInput::with_default_params(close_prices);
        b.iter(|| calculate_gaussian(black_box(&input)).expect("Failed to calculate GAUSSIAN"))
    });

    // Super Smoother 3 Pole
    group.bench_function(BenchmarkId::new("SUPERSMOOTHER3POLE", 0), |b| {
        let input = SuperSmoother3PoleInput::with_default_params(close_prices);
        b.iter(|| {
            calculate_supersmoother_3_pole(black_box(&input))
                .expect("Failed to calculate SUPERSMOOTHER3POLE")
        })
    });

    // Super Smoother
    group.bench_function(BenchmarkId::new("SUPERSMOOTHER", 0), |b| {
        let input = SuperSmootherInput::with_default_params(close_prices);
        b.iter(|| {
            calculate_supersmoother(black_box(&input)).expect("Failed to calculate SUPERSMOOTHER")
        })
    });

    // SinWMA
    group.bench_function(BenchmarkId::new("SINWMA", 0), |b| {
        let input = SinWmaInput::with_default_params(close_prices);
        b.iter(|| calculate_sinwma(black_box(&input)).expect("Failed to calculate SINWMA"))
    });

    // Wilders
    group.bench_function(BenchmarkId::new("WILDERS", 0), |b| {
        let input = WildersInput::with_default_params(close_prices);
        b.iter(|| calculate_wilders(black_box(&input)).expect("Failed to calculate WILDERS"))
    });

    // Linear Regression
    group.bench_function(BenchmarkId::new("LINREG", 0), |b| {
        let input = LinRegInput::with_default_params(close_prices);
        b.iter(|| calculate_linreg(black_box(&input)).expect("Failed to calculate LINREG"))
    });

    // HMA
    group.bench_function(BenchmarkId::new("HMA", 0), |b| {
        let input = HmaInput::with_default_params(close_prices);
        b.iter(|| calculate_hma(black_box(&input)).expect("Failed to calculate HMA"))
    });

    // FWMA
    group.bench_function(BenchmarkId::new("FWMA", 0), |b| {
        let input = FwmaInput::with_default_params(close_prices);
        b.iter(|| calculate_fwma(black_box(&input)).expect("Failed to calculate FWMA"))
    });

    // MAMA
    group.bench_function(BenchmarkId::new("MAMA", 0), |b| {
        let input = MamaInput::with_default_params(close_prices);
        b.iter(|| calculate_mama(black_box(&input)).expect("Failed to calculate MAMA"))
    });

    // T3
    group.bench_function(BenchmarkId::new("T3", 0), |b| {
        let input = T3Input::with_default_params(close_prices);
        b.iter(|| calculate_t3(black_box(&input)).expect("Failed to calculate T3"))
    });

    // KAMA
    group.bench_function(BenchmarkId::new("KAMA", 0), |b| {
        let input = KamaInput::with_default_params(close_prices);
        b.iter(|| calculate_kama(black_box(&input)).expect("Failed to calculate KAMA"))
    });

    // TRIMA
    group.bench_function(BenchmarkId::new("TRIMA", 0), |b| {
        let input = TrimaInput::with_default_params(close_prices);
        b.iter(|| calculate_trima(black_box(&input)).expect("Failed to calculate TRIMA"))
    });

    // TEMA
    group.bench_function(BenchmarkId::new("TEMA", 0), |b| {
        let input = TemaInput::with_default_params(close_prices);
        b.iter(|| calculate_tema(black_box(&input)).expect("Failed to calculate TEMA"))
    });

    // DEMA
    group.bench_function(BenchmarkId::new("DEMA", 0), |b| {
        let input = DemaInput::with_default_params(close_prices);
        b.iter(|| calculate_dema(black_box(&input)).expect("Failed to calculate DEMA"))
    });

    // WMA
    group.bench_function(BenchmarkId::new("WMA", 0), |b| {
        let input = WmaInput::with_default_params(close_prices);
        b.iter(|| calculate_wma(black_box(&input)).expect("Failed to calculate WMA"))
    });

    // BANDPASS
    group.bench_function(BenchmarkId::new("BANDPASS", 0), |b| {
        let input = BandPassInput::with_default_params(close_prices);
        b.iter(|| calculate_bandpass(black_box(&input)).expect("Failed to calculate BANDPASS"))
    });

    // HIGHPASS
    group.bench_function(BenchmarkId::new("HIGHPASS", 0), |b| {
        let input = HighPassInput::with_default_params(close_prices);
        b.iter(|| calculate_highpass(black_box(&input)).expect("Failed to calculate HIGHPASS"))
    });

    // AVGPRICE
    group.bench_function(BenchmarkId::new("AVGPRICE", 0), |b| {
        let input = AvgPriceInput::with_default_params(&candles);
        b.iter(|| calculate_avgprice(&input).expect("Failed to calculate AVGPRICE"))
    });

    // ATR
    group.bench_function(BenchmarkId::new("ATR", 0), |b| {
        let input = AtrInput::with_default_params(&candles);
        b.iter(|| calculate_atr(black_box(&input)).expect("Failed to calculate ATR"))
    });

    // AROONOSC
    group.bench_function(BenchmarkId::new("AROONOSC", 0), |b| {
        let input = AroonOscInput::with_default_params(&candles);
        b.iter(|| calculate_aroon_osc(black_box(&input)).expect("Failed to calculate AROONOSC"))
    });

    // AROON
    group.bench_function(BenchmarkId::new("AROON", 0), |b| {
        let input = AroonInput::with_default_params(&candles);
        b.iter(|| calculate_aroon(black_box(&input)).expect("Failed to calculate AROON"))
    });

    // APO
    group.bench_function(BenchmarkId::new("APO", 0), |b| {
        let input = ApoInput::with_default_params(&candles);
        b.iter(|| calculate_apo(black_box(&input)).expect("Failed to calculate APO"))
    });

    // AO
    group.bench_function(BenchmarkId::new("AO", 0), |b| {
        let input = AoInput::with_default_params(&hl2_prices);
        b.iter(|| calculate_ao(black_box(&input)).expect("Failed to calculate AO"))
    });

    // ALMA
    group.bench_function(BenchmarkId::new("ALMA", 0), |b| {
        let input = AlmaInput::with_default_params(close_prices);
        b.iter(|| calculate_alma(black_box(&input)).expect("Failed to calculate ALMA"))
    });

    // ADOSC
    group.bench_function(BenchmarkId::new("ADOSC", 0), |b| {
        let input = AdoscInput::with_default_params(&candles);
        b.iter(|| calculate_adosc(black_box(&input)).expect("Failed to calculate ADOSC"))
    });

    // ZLEMA
    group.bench_function(BenchmarkId::new("ZLEMA", 0), |b| {
        let input = ZlemaInput::with_default_params(close_prices);
        b.iter(|| calculate_zlema(black_box(&input)).expect("Failed to calculate ZLEMA"))
    });

    // Alligator
    group.bench_function(BenchmarkId::new("ALLIGATOR", 0), |b| {
        let input = AlligatorInput::with_default_params(&hl2_prices);
        b.iter(|| calculate_alligator(black_box(&input)).expect("Failed to calculate alligator"))
    });

    // ADXR
    group.bench_function(BenchmarkId::new("ADXR", 0), |b| {
        let input = AdxrInput::with_default_params(&candles);
        b.iter(|| calculate_adxr(black_box(&input)).expect("Failed to calculate ADXR"))
    });

    // ADX
    group.bench_function(BenchmarkId::new("ADX", 0), |b| {
        let input = AdxInput::with_default_params(&candles);
        b.iter(|| calculate_adx(black_box(&input)).expect("Failed to calculate ADX"))
    });

    // SMA
    group.bench_function(BenchmarkId::new("SMA", 0), |b| {
        let input = SmaInput::with_default_params(close_prices);
        b.iter(|| calculate_sma(black_box(&input)).expect("Failed to calculate SMA"))
    });

    // EMA
    group.bench_function(BenchmarkId::new("EMA", 0), |b| {
        let input = EmaInput::with_default_params(close_prices);
        b.iter(|| calculate_ema(black_box(&input)).expect("Failed to calculate EMA"))
    });

    // RSI
    group.bench_function(BenchmarkId::new("RSI", 0), |b| {
        let input = RsiInput::with_default_params(close_prices);
        b.iter(|| calculate_rsi(black_box(&input)).expect("Failed to calculate RSI"))
    });

    // ACOSC
    group.bench_function(BenchmarkId::new("ACOSC", 0), |b| {
        let input = AcoscInput::with_default_params(&candles);
        b.iter(|| calculate_acosc(black_box(&input)).expect("Failed to calculate ACOSC"))
    });

    // AD
    group.bench_function(BenchmarkId::new("AD", 0), |b| {
        let input = AdInput::with_default_params(&candles);
        b.iter(|| calculate_ad(black_box(&input)).expect("Failed to calculate AD"))
    });

    group.finish();
}

criterion_group!(benches, benchmark_indicators);
criterion_main!(benches);
