extern crate criterion;
extern crate my_project;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use my_project::utilities::data_loader::read_candles_from_csv;

use my_project::indicators::{
    acosc::{acosc, AcoscInput},
    ad::{ad, AdInput},
    adosc::{adosc, AdoscInput},
    adx::{adx, AdxInput},
    adxr::{adxr, AdxrInput},
    alligator::{alligator, AlligatorInput},
    alma::{alma, AlmaInput},
    ao::{ao, AoInput},
    apo::{apo, ApoInput},
    aroon::{aroon, AroonInput},
    aroonosc::{aroon_osc, AroonOscInput},
    atr::{atr, AtrInput},
    avgprice::{avgprice, AvgPriceInput},
    bandpass::{bandpass, BandPassInput},
    bollinger_bands::{bollinger_bands, BollingerBandsInput},
    bollinger_bands_width::{bollinger_bands_width, BollingerBandsWidthInput},
    bop::{bop, BopInput},
    cci::{cci, CciInput},
    cwma::{cwma, CwmaInput},
    dema::{dema, DemaInput},
    edcf::{edcf, EdcfInput},
    ehlers_itrend::{ehlers_itrend, EhlersITrendInput},
    ema::{ema, EmaInput},
    epma::{epma, EpmaInput},
    fwma::{fwma, FwmaInput},
    gaussian::{gaussian, GaussianInput},
    highpass::{highpass, HighPassInput},
    highpass_2_pole::{highpass_2_pole, HighPass2Input},
    hma::{hma, HmaInput},
    hwma::{hwma, HwmaInput},
    jma::{jma, JmaInput},
    jsa::{jsa, JsaInput},
    kama::{kama, KamaInput},
    linreg::{linreg, LinRegInput},
    maaq::{maaq, MaaqInput},
    mama::{mama, MamaInput},
    mwdx::{mwdx, MwdxInput},
    nma::{nma, NmaInput},
    pwma::{pwma, PwmaInput},
    reflex::{reflex, ReflexInput},
    roc::{roc, RocInput},
    rocp::{rocp, RocpInput},
    rsi::{rsi, RsiInput},
    sinwma::{sinwma, SinWmaInput},
    sma::{sma, SmaInput},
    smma::{smma, SmmaInput},
    sqwma::{sqwma, SqwmaInput},
    srwma::{srwma, SrwmaInput},
    supersmoother::{supersmoother, SuperSmootherInput},
    supersmoother_3_pole::{supersmoother_3_pole, SuperSmoother3PoleInput},
    swma::{swma, SwmaInput},
    tema::{tema, TemaInput},
    tilson::{tilson, TilsonInput},
    trendflex::{trendflex, TrendFlexInput},
    trima::{trima, TrimaInput},
    vpwma::{vpwma, VpwmaInput},
    vwap::{vwap, VwapInput},
    vwma::{vwma, VwmaInput},
    wilders::{wilders, WildersInput},
    wma::{wma, WmaInput},
    zlema::{zlema, ZlemaInput},
};
use std::time::Duration;

fn benchmark_indicators(c: &mut Criterion) {
    let candles =
        read_candles_from_csv("src/data/bitfinex btc-usd 100,000 candles ends 09-01-24.csv")
            .expect("Failed to load candles");

    let mut group = c.benchmark_group("Indicator Benchmarks");
    group.measurement_time(Duration::new(8, 0));
    group.warm_up_time(Duration::new(4, 0));

    // Bollinger Bands Width
    group.bench_function(BenchmarkId::new("BOLLINGER_BANDS_WIDTH", 0), |b| {
        let input = BollingerBandsWidthInput::with_default_candles(&candles);
        b.iter(|| {
            bollinger_bands_width(black_box(&input))
                .expect("Failed to calculate BOLLINGER_BANDS_WIDTH")
        })
    });

    // ROCP
    group.bench_function(BenchmarkId::new("ROCP", 0), |b| {
        let input = RocpInput::with_default_candles(&candles);
        b.iter(|| rocp(black_box(&input)).expect("Failed to calculate ROCP"))
    });

    // BOP
    group.bench_function(BenchmarkId::new("BOP", 0), |b| {
        let input = BopInput::with_default_candles(&candles);
        b.iter(|| bop(black_box(&input)).expect("Failed to calculate BOP"))
    });

    // CCI
    group.bench_function(BenchmarkId::new("CCI", 0), |b| {
        let input = CciInput::with_default_candles(&candles);
        b.iter(|| cci(black_box(&input)).expect("Failed to calculate CCI"))
    });

    // Bollinger Bands
    group.bench_function(BenchmarkId::new("BOLLINGER_BANDS", 0), |b| {
        let input = BollingerBandsInput::with_default_candles(&candles);
        b.iter(|| bollinger_bands(black_box(&input)).expect("Failed to calculate BOLLINGER_BANDS"))
    });

    // ROC
    group.bench_function(BenchmarkId::new("ROC", 0), |b| {
        let input = RocInput::with_default_candles(&candles);
        b.iter(|| roc(black_box(&input)).expect("Failed to calculate ROC"))
    });

    // EPMA
    group.bench_function(BenchmarkId::new("EPMA", 0), |b| {
        let input = EpmaInput::with_default_candles(&candles);
        b.iter(|| epma(black_box(&input)).expect("Failed to calculate EPMA"))
    });

    // JSA
    group.bench_function(BenchmarkId::new("JSA", 0), |b| {
        let input = JsaInput::with_default_candles(&candles);
        b.iter(|| jsa(black_box(&input)).expect("Failed to calculate JSA"))
    });

    // CWMA
    group.bench_function(BenchmarkId::new("CWMA", 0), |b| {
        let input = CwmaInput::with_default_candles(&candles);
        b.iter(|| cwma(black_box(&input)).expect("Failed to calculate CWMA"))
    });

    // VPWMA
    group.bench_function(BenchmarkId::new("VPWMA", 0), |b| {
        let input = VpwmaInput::with_default_candles(&candles);
        b.iter(|| vpwma(black_box(&input)).expect("Failed to calculate VPWMA"))
    });

    // SRWMA
    group.bench_function(BenchmarkId::new("SRWMA", 0), |b| {
        let input = SrwmaInput::with_default_candles(&candles);
        b.iter(|| srwma(black_box(&input)).expect("Failed to calculate SRWMA"))
    });

    // SQWMA
    group.bench_function(BenchmarkId::new("SQWMA", 0), |b| {
        let input = SqwmaInput::with_default_candles(&candles);
        b.iter(|| sqwma(black_box(&input)).expect("Failed to calculate SQWMA"))
    });

    // MAAQ
    group.bench_function(BenchmarkId::new("MAAQ", 0), |b| {
        let input = MaaqInput::with_default_candles(&candles);
        b.iter(|| maaq(black_box(&input)).expect("Failed to calculate MAAQ"))
    });

    // MWDX
    group.bench_function(BenchmarkId::new("MWDX", 0), |b| {
        let input = MwdxInput::with_default_candles(&candles);
        b.iter(|| mwdx(black_box(&input)).expect("Failed to calculate MWDX"))
    });

    // NMA
    group.bench_function(BenchmarkId::new("NMA", 0), |b| {
        let input = NmaInput::with_default_candles(&candles);
        b.iter(|| nma(black_box(&input)).expect("Failed to calculate NMA"))
    });

    // EDCF
    group.bench_function(BenchmarkId::new("EDCF", 0), |b| {
        let input = EdcfInput::with_default_candles(&candles);
        b.iter(|| edcf(black_box(&input)).expect("Failed to calculate EDCF"))
    });

    // VWAP
    group.bench_function(BenchmarkId::new("VWAP", 0), |b| {
        let input = VwapInput::with_default_candles(&candles);
        b.iter(|| vwap(black_box(&input)).expect("Failed to calculate VWAP"))
    });

    // HWMA
    group.bench_function(BenchmarkId::new("HWMA", 0), |b| {
        let input = HwmaInput::with_default_candles(&candles);
        b.iter(|| hwma(black_box(&input)).expect("Failed to calculate HWMA"))
    });

    // SWMA
    group.bench_function(BenchmarkId::new("SWMA", 0), |b| {
        let input = SwmaInput::with_default_candles(&candles);
        b.iter(|| swma(black_box(&input)).expect("Failed to calculate SWMA"))
    });

    // TrendFlex
    group.bench_function(BenchmarkId::new("TRENDFLEX", 0), |b| {
        let input = TrendFlexInput::with_default_candles(&candles);
        b.iter(|| trendflex(black_box(&input)).expect("Failed to calculate TRENDFLEX"))
    });

    // VWMA
    group.bench_function(BenchmarkId::new("VWMA", 0), |b| {
        let input = VwmaInput::with_default_candles(&candles);
        b.iter(|| vwma(black_box(&input)).expect("Failed to calculate VWMA"))
    });

    // PWMA
    group.bench_function(BenchmarkId::new("PWMA", 0), |b| {
        let input = PwmaInput::with_default_candles(&candles);
        b.iter(|| pwma(black_box(&input)).expect("Failed to calculate PWMA"))
    });

    // ITREND
    group.bench_function(BenchmarkId::new("ITREND", 0), |b| {
        let input = EhlersITrendInput::with_default_candles(&candles);
        b.iter(|| ehlers_itrend(black_box(&input)).expect("Failed to calculate Ehler's ITrend"))
    });

    // SMMA
    group.bench_function(BenchmarkId::new("SMMA", 0), |b| {
        let input = SmmaInput::with_default_candles(&candles);
        b.iter(|| smma(black_box(&input)).expect("Failed to calculate SMMA"))
    });

    // Reflex
    group.bench_function(BenchmarkId::new("REFLEX", 0), |b| {
        let input = ReflexInput::with_default_candles(&candles);
        b.iter(|| reflex(black_box(&input)).expect("Failed to calculate REFLEX"))
    });

    // JMA
    group.bench_function(BenchmarkId::new("JMA", 0), |b| {
        let input = JmaInput::with_default_candles(&candles);
        b.iter(|| jma(black_box(&input)).expect("Failed to calculate JMA"))
    });

    // High Pass 2 Pole
    group.bench_function(BenchmarkId::new("HIGHPASS_2Pole", 0), |b| {
        let input = HighPass2Input::with_default_candles(&candles);
        b.iter(|| highpass_2_pole(black_box(&input)).expect("Failed to calculate HIGHPASS2"))
    });

    // High Pass
    group.bench_function(BenchmarkId::new("HIGHPASS_1Pole", 0), |b| {
        let input = HighPassInput::with_default_candles(&candles);
        b.iter(|| highpass(black_box(&input)).expect("Failed to calculate HIGHPASS"))
    });

    // Gaussian
    group.bench_function(BenchmarkId::new("GAUSSIAN", 0), |b| {
        let input = GaussianInput::with_default_candles(&candles);
        b.iter(|| gaussian(black_box(&input)).expect("Failed to calculate GAUSSIAN"))
    });

    // Super Smoother 3 Pole
    group.bench_function(BenchmarkId::new("SUPERSMOOTHER3POLE", 0), |b| {
        let input = SuperSmoother3PoleInput::with_default_candles(&candles);
        b.iter(|| {
            supersmoother_3_pole(black_box(&input)).expect("Failed to calculate SUPERSMOOTHER3POLE")
        })
    });

    // Super Smoother
    group.bench_function(BenchmarkId::new("SUPERSMOOTHER", 0), |b| {
        let input = SuperSmootherInput::with_default_candles(&candles);
        b.iter(|| supersmoother(black_box(&input)).expect("Failed to calculate SUPERSMOOTHER"))
    });

    // SinWMA
    group.bench_function(BenchmarkId::new("SINWMA", 0), |b| {
        let input = SinWmaInput::with_default_candles(&candles);
        b.iter(|| sinwma(black_box(&input)).expect("Failed to calculate SINWMA"))
    });

    // Wilders
    group.bench_function(BenchmarkId::new("WILDERS", 0), |b| {
        let input = WildersInput::with_default_candles(&candles);
        b.iter(|| wilders(black_box(&input)).expect("Failed to calculate WILDERS"))
    });

    // Linear Regression
    group.bench_function(BenchmarkId::new("LINREG", 0), |b| {
        let input = LinRegInput::with_default_candles(&candles);
        b.iter(|| linreg(black_box(&input)).expect("Failed to calculate LINREG"))
    });

    // HMA
    group.bench_function(BenchmarkId::new("HMA", 0), |b| {
        let input = HmaInput::with_default_candles(&candles);
        b.iter(|| hma(black_box(&input)).expect("Failed to calculate HMA"))
    });

    // FWMA
    group.bench_function(BenchmarkId::new("FWMA", 0), |b| {
        let input = FwmaInput::with_default_candles(&candles);
        b.iter(|| fwma(black_box(&input)).expect("Failed to calculate FWMA"))
    });

    // MAMA
    group.bench_function(BenchmarkId::new("MAMA", 0), |b| {
        let input = MamaInput::with_default_candles(&candles);
        b.iter(|| mama(black_box(&input)).expect("Failed to calculate MAMA"))
    });

    // TILSON
    group.bench_function(BenchmarkId::new("TILSON", 0), |b| {
        let input = TilsonInput::with_default_candles(&candles);
        b.iter(|| tilson(black_box(&input)).expect("Failed to calculate T3"))
    });

    // KAMA
    group.bench_function(BenchmarkId::new("KAMA", 0), |b| {
        let input = KamaInput::with_default_candles(&candles);
        b.iter(|| kama(black_box(&input)).expect("Failed to calculate KAMA"))
    });

    // TRIMA
    group.bench_function(BenchmarkId::new("TRIMA", 0), |b| {
        let input = TrimaInput::with_default_candles(&candles);
        b.iter(|| trima(black_box(&input)).expect("Failed to calculate TRIMA"))
    });

    // TEMA
    group.bench_function(BenchmarkId::new("TEMA", 0), |b| {
        let input = TemaInput::with_default_candles(&candles);
        b.iter(|| tema(black_box(&input)).expect("Failed to calculate TEMA"))
    });

    // DEMA
    group.bench_function(BenchmarkId::new("DEMA", 0), |b| {
        let input = DemaInput::with_default_candles(&candles);
        b.iter(|| dema(black_box(&input)).expect("Failed to calculate DEMA"))
    });

    // WMA
    group.bench_function(BenchmarkId::new("WMA", 0), |b| {
        let input = WmaInput::with_default_candles(&candles);
        b.iter(|| wma(black_box(&input)).expect("Failed to calculate WMA"))
    });

    // BANDPASS
    group.bench_function(BenchmarkId::new("BANDPASS", 0), |b| {
        let input = BandPassInput::with_default_candles(&candles);
        b.iter(|| bandpass(black_box(&input)).expect("Failed to calculate BANDPASS"))
    });

    // HIGHPASS
    group.bench_function(BenchmarkId::new("HIGHPASS", 0), |b| {
        let input = HighPassInput::with_default_candles(&candles);
        b.iter(|| highpass(black_box(&input)).expect("Failed to calculate HIGHPASS"))
    });

    // AVGPRICE
    group.bench_function(BenchmarkId::new("AVGPRICE", 0), |b| {
        let input = AvgPriceInput::with_default_candles(&candles);
        b.iter(|| avgprice(&input).expect("Failed to calculate AVGPRICE"))
    });

    // ATR
    group.bench_function(BenchmarkId::new("ATR", 0), |b| {
        let input = AtrInput::with_default_candles(&candles);
        b.iter(|| atr(black_box(&input)).expect("Failed to calculate ATR"))
    });

    // AROONOSC
    group.bench_function(BenchmarkId::new("AROONOSC", 0), |b| {
        let input = AroonOscInput::with_default_candles(&candles);
        b.iter(|| aroon_osc(black_box(&input)).expect("Failed to calculate AROONOSC"))
    });

    // AROON
    group.bench_function(BenchmarkId::new("AROON", 0), |b| {
        let input = AroonInput::with_default_candles(&candles);
        b.iter(|| aroon(black_box(&input)).expect("Failed to calculate AROON"))
    });

    // APO
    group.bench_function(BenchmarkId::new("APO", 0), |b| {
        let input = ApoInput::with_default_candles(&candles);
        b.iter(|| apo(black_box(&input)).expect("Failed to calculate APO"))
    });

    // AO
    group.bench_function(BenchmarkId::new("AO", 0), |b| {
        let input = AoInput::with_default_candles(&candles);
        b.iter(|| ao(black_box(&input)).expect("Failed to calculate AO"))
    });

    // ALMA
    group.bench_function(BenchmarkId::new("ALMA", 0), |b| {
        let input = AlmaInput::with_default_candles(&candles);
        b.iter(|| alma(black_box(&input)).expect("Failed to calculate ALMA"))
    });

    // ADOSC
    group.bench_function(BenchmarkId::new("ADOSC", 0), |b| {
        let input = AdoscInput::with_default_candles(&candles);
        b.iter(|| adosc(black_box(&input)).expect("Failed to calculate ADOSC"))
    });

    // ZLEMA
    group.bench_function(BenchmarkId::new("ZLEMA", 0), |b| {
        let input = ZlemaInput::with_default_candles(&candles);
        b.iter(|| zlema(black_box(&input)).expect("Failed to calculate ZLEMA"))
    });

    // Alligator
    group.bench_function(BenchmarkId::new("ALLIGATOR", 0), |b| {
        let input = AlligatorInput::with_default_candles(&candles);
        b.iter(|| alligator(black_box(&input)).expect("Failed to calculate alligator"))
    });

    // ADXR
    group.bench_function(BenchmarkId::new("ADXR", 0), |b| {
        let input = AdxrInput::with_default_candles(&candles);
        b.iter(|| adxr(black_box(&input)).expect("Failed to calculate ADXR"))
    });

    // ADX
    group.bench_function(BenchmarkId::new("ADX", 0), |b| {
        let input = AdxInput::with_default_candles(&candles);
        b.iter(|| adx(black_box(&input)).expect("Failed to calculate ADX"))
    });

    // SMA
    group.bench_function(BenchmarkId::new("SMA", 0), |b| {
        let input = SmaInput::with_default_candles(&candles);
        b.iter(|| sma(black_box(&input)).expect("Failed to calculate SMA"))
    });

    // EMA
    group.bench_function(BenchmarkId::new("EMA", 0), |b| {
        let input = EmaInput::with_default_candles(&candles);
        b.iter(|| ema(black_box(&input)).expect("Failed to calculate EMA"))
    });

    // RSI
    group.bench_function(BenchmarkId::new("RSI", 0), |b| {
        let input = RsiInput::with_default_candles(&candles);
        b.iter(|| rsi(black_box(&input)).expect("Failed to calculate RSI"))
    });

    // ACOSC
    group.bench_function(BenchmarkId::new("ACOSC", 0), |b| {
        let input = AcoscInput::with_default_candles(&candles);
        b.iter(|| acosc(black_box(&input)).expect("Failed to calculate ACOSC"))
    });

    // AD
    group.bench_function(BenchmarkId::new("AD", 0), |b| {
        let input = AdInput::with_default_candles(&candles);
        b.iter(|| ad(black_box(&input)).expect("Failed to calculate AD"))
    });

    group.finish();
}

criterion_group!(benches, benchmark_indicators);
criterion_main!(benches);
