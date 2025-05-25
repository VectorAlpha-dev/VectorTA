use anyhow::anyhow;
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion, Throughput,
};
use my_project::utilities::enums::Kernel;
use once_cell::sync::Lazy;
use paste::paste;
use std::time::{Duration, Instant};

use my_project::indicators::{
    acosc::{acosc as acosc_raw, AcoscInput},
    ad::{ad as ad_raw, AdInput},
    adosc::{adosc as adosc_raw, AdoscInput},
    adx::{adx as adx_raw, AdxInput},
    adxr::{adxr as adxr_raw, AdxrInput},
    alligator::{alligator as alligator_raw, AlligatorInput},
    alma::{alma_with_kernel, AlmaBatchBuilder, AlmaData, AlmaInput},
    ao::{ao as ao_raw, AoInput},
    apo::{apo as apo_raw, ApoInput},
    aroon::{aroon as aroon_raw, AroonInput},
    aroonosc::{aroon_osc as aroon_osc_raw, AroonOscInput},
    atr::{atr as atr_raw, AtrInput},
    avgprice::{avgprice as avgprice_raw, AvgPriceInput},
    bandpass::{bandpass as bandpass_raw, BandPassInput},
    bollinger_bands::{bollinger_bands as bollinger_bands_raw, BollingerBandsInput},
    bollinger_bands_width::{
        bollinger_bands_width as bollinger_bands_width_raw, BollingerBandsWidthInput,
    },
    bop::{bop as bop_raw, BopInput},
    cci::{cci as cci_raw, CciInput},
    cfo::{cfo as cfo_raw, CfoInput},
    cg::{cg as cg_raw, CgInput},
    chande::{chande as chande_raw, ChandeInput},
    chop::{chop as chop_raw, ChopInput},
    cksp::{cksp as cksp_raw, CkspInput},
    cmo::{cmo as cmo_raw, CmoInput},
    coppock::{coppock as coppock_raw, CoppockInput},
    correl_hl::{correl_hl as correl_hl_raw, CorrelHlInput},
    correlation_cycle::{correlation_cycle as correlation_cycle_raw, CorrelationCycleInput},
    cvi::{cvi as cvi_raw, CviInput},
    cwma::{cwma_with_kernel, CwmaBatchBuilder, CwmaData, CwmaInput},
    damiani_volatmeter::{damiani_volatmeter as damiani_volatmeter_raw, DamianiVolatmeterInput},
    dec_osc::{dec_osc as dec_osc_raw, DecOscInput},
    decycler::{decycler as decycler_raw, DecyclerInput},
    dema::{dema_with_kernel, DemaBatchBuilder, DemaData, DemaInput},
    devstop::{devstop as devstop_raw, DevStopInput},
    di::{di as di_raw, DiInput},
    dm::{dm as dm_raw, DmInput},
    donchian::{donchian as donchian_raw, DonchianInput},
    dpo::{dpo as dpo_raw, DpoInput},
    dti::{dti as dti_raw, DtiInput},
    dx::{dx as dx_raw, DxInput},
    edcf::{edcf_with_kernel, EdcfBatchBuilder, EdcfData, EdcfInput},
    efi::{efi as efi_raw, EfiInput},
    ehlers_itrend::{
        ehlers_itrend_with_kernel, EhlersITrendBatchBuilder, EhlersITrendData, EhlersITrendInput,
    },
    ema::{ema_with_kernel, EmaBatchBuilder, EmaData, EmaInput},
    emd::{emd as emd_raw, EmdInput},
    emv::{emv as emv_raw, EmvInput},
    epma::{epma_with_kernel, EpmaBatchBuilder, EpmaData, EpmaInput},
    er::{er as er_raw, ErInput},
    eri::{eri as eri_raw, EriInput},
    fisher::{fisher as fisher_raw, FisherInput},
    fosc::{fosc as fosc_raw, FoscInput},
    frama::{frama_with_kernel, FramaBatchBuilder, FramaData, FramaInput},
    fwma::{fwma_with_kernel, FwmaBatchBuilder, FwmaData, FwmaInput},
    gatorosc::{gatorosc as gatorosc_raw, GatorOscInput},
    gaussian::{gaussian as gaussian_raw, GaussianInput},
    heikin_ashi_candles::{heikin_ashi_candles as heikin_ashi_candles_raw, HeikinAshiInput},
    highpass::{highpass as highpass_raw, HighPassInput},
    highpass_2_pole::{highpass_2_pole as highpass_2_pole_raw, HighPass2Input},
    hma::{hma as hma_raw, HmaInput},
    ht_dcperiod::{ht_dcperiod as ht_dcperiod_raw, HtDcPeriodInput},
    hwma::{hwma as hwma_raw, HwmaInput},
    ift_rsi::{ift_rsi as ift_rsi_raw, IftRsiInput},
    jma::{jma as jma_raw, JmaInput},
    jsa::{jsa as jsa_raw, JsaInput},
    kama::{kama as kama_raw, KamaInput},
    kaufmanstop::{kaufmanstop as kaufmanstop_raw, KaufmanstopInput},
    kdj::{kdj as kdj_raw, KdjInput},
    keltner::{keltner as keltner_raw, KeltnerInput},
    kst::{kst as kst_raw, KstInput},
    kurtosis::{kurtosis as kurtosis_raw, KurtosisInput},
    kvo::{kvo as kvo_raw, KvoInput},
    linearreg_angle::{linearreg_angle as linearreg_angle_raw, Linearreg_angleInput},
    linearreg_intercept::{
        linearreg_intercept as linearreg_intercept_raw, LinearRegInterceptInput,
    },
    linearreg_slope::{linearreg_slope as linearreg_slope_raw, LinearRegSlopeInput},
    linreg::{linreg as linreg_raw, LinRegInput},
    lrsi::{lrsi as lrsi_raw, LrsiInput},
    maaq::{maaq as maaq_raw, MaaqInput},
    mab::{mab as mab_raw, MabInput},
    macd::{macd as macd_raw, MacdInput},
    mama::{mama as mama_raw, MamaInput},
    marketefi::{marketfi as marketfi_raw, MarketefiInput},
    mass::{mass as mass_raw, MassInput},
    mean_ad::{mean_ad as mean_ad_raw, MeanAdInput},
    medium_ad::{medium_ad as medium_ad_raw, MediumAdInput},
    medprice::{medprice as medprice_raw, MedpriceInput},
    mfi::{mfi as mfi_raw, MfiInput},
    midpoint::{midpoint as midpoint_raw, MidpointInput},
    midprice::{midprice as midprice_raw, MidpriceInput},
    minmax::{minmax as minmax_raw, MinmaxInput},
    mom::{mom as mom_raw, MomInput},
    msw::{msw as msw_raw, MswInput},
    mwdx::{mwdx as mwdx_raw, MwdxInput},
    natr::{natr as natr_raw, NatrInput},
    nma::{nma as nma_raw, NmaInput},
    pivot::{pivot as pivot_raw, PivotInput},
    pma::{pma as pma_raw, PmaInput},
    ppo::{ppo as ppo_raw, PpoInput},
    pvi::{pvi as pvi_raw, PviInput},
    pwma::{pwma as pwma_raw, PwmaInput},
    qstick::{qstick as qstick_raw, QstickInput},
    reflex::{reflex as reflex_raw, ReflexInput},
    roc::{roc as roc_raw, RocInput},
    rocp::{rocp as rocp_raw, RocpInput},
    rocr::{rocr as rocr_raw, RocrInput},
    rsi::{rsi as rsi_raw, RsiInput},
    rvi::{rvi as rvi_raw, RviInput},
    safezonestop::{safezonestop as safezonestop_raw, SafeZoneStopInput},
    sar::{sar as sar_raw, SarInput},
    sinwma::{sinwma as sinwma_raw, SinWmaInput},
    sma::{sma as sma_raw, SmaInput},
    smma::{smma as smma_raw, SmmaInput},
    squeeze_momentum::{squeeze_momentum as squeeze_momentum_raw, SqueezeMomentumInput},
    sqwma::{sqwma as sqwma_raw, SqwmaInput},
    srsi::{srsi as srsi_raw, SrsiInput},
    srwma::{srwma as srwma_raw, SrwmaInput},
    stc::{stc as stc_raw, StcInput},
    stddev::{stddev as stddev_raw, StdDevInput},
    stochf::{stochf as stochf_raw, StochfInput},
    supersmoother::{supersmoother as supersmoother_raw, SuperSmootherInput},
    supersmoother_3_pole::{
        supersmoother_3_pole as supersmoother_3_pole_raw, SuperSmoother3PoleInput,
    },
    swma::{swma as swma_raw, SwmaInput},
    tema::{tema as tema_raw, TemaInput},
    tilson::{tilson as tilson_raw, TilsonInput},
    trendflex::{trendflex as trendflex_raw, TrendFlexInput},
    trima::{trima as trima_raw, TrimaInput},
    trix::{trix as trix_raw, TrixInput},
    tsi::{tsi as tsi_raw, TsiInput},
    ttm_trend::{ttm_trend as ttm_trend_raw, TtmTrendInput},
    ui::{ui as ui_raw, UiInput},
    ultosc::{ultosc as ultosc_raw, UltOscInput},
    var::{var as var_raw, VarInput},
    vi::{vi as vi_raw, ViInput},
    vidya::{vidya as vidya_raw, VidyaInput},
    vlma::{vlma as vlma_raw, VlmaInput},
    vosc::{vosc as vosc_raw, VoscInput},
    voss::{voss as voss_raw, VossInput},
    vpci::{vpci as vpci_raw, VpciInput},
    vpt::{vpt as vpt_raw, VptInput},
    vpwma::{vpwma as vpwma_raw, VpwmaInput},
    vwap::{vwap as vwap_raw, VwapInput},
    vwma::{vwma as vwma_raw, VwmaInput},
    vwmacd::{vwmacd as vwmacd_raw, VwmacdInput},
    wad::{wad as wad_raw, WadInput},
    wavetrend::{wavetrend as wavetrend_raw, WavetrendInput},
    wclprice::{wclprice as wclprice_raw, WclpriceInput},
    wilders::{wilders as wilders_raw, WildersInput},
    willr::{willr as willr_raw, WillrInput},
    wma::{wma as wma_raw, WmaInput},
    zlema::{zlema as zlema_raw, ZlemaInput},
    zscore::{zscore as zscore_raw, ZscoreInput},
};

use my_project::utilities::data_loader::{read_candles_from_csv, Candles};

static CANDLES_10K: Lazy<Candles> =
    Lazy::new(|| read_candles_from_csv("src/data/10kCandles.csv").expect("10 k candles csv"));
static CANDLES_100K: Lazy<Candles> = Lazy::new(|| {
    read_candles_from_csv("src/data/bitfinex btc-usd 100,000 candles ends 09-01-24.csv")
        .expect("100 k candles csv")
});
static CANDLES_1M: Lazy<Candles> =
    Lazy::new(|| read_candles_from_csv("src/data/1MillionCandles.csv").expect("1 M candles csv"));

trait InputLen {
    fn with_len(len: usize) -> Self;
}

pub type AcoscInputS = AcoscInput<'static>;
pub type AdInputS = AdInput<'static>;
pub type AdoscInputS = AdoscInput<'static>;
pub type AdxInputS = AdxInput<'static>;
pub type AdxrInputS = AdxrInput<'static>;
pub type AlligatorInputS = AlligatorInput<'static>;
pub type AlmaInputS = AlmaInput<'static>;
pub type AoInputS = AoInput<'static>;
pub type ApoInputS = ApoInput<'static>;
pub type AroonInputS = AroonInput<'static>;
pub type AroonOscInputS = AroonOscInput<'static>;
pub type AtrInputS = AtrInput<'static>;
pub type AvgPriceInputS = AvgPriceInput<'static>;
pub type BandPassInputS = BandPassInput<'static>;
pub type BollingerBandsInputS = BollingerBandsInput<'static>;
pub type BollingerBandsWidthInputS = BollingerBandsWidthInput<'static>;
pub type BopInputS = BopInput<'static>;
pub type CciInputS = CciInput<'static>;
pub type CfoInputS = CfoInput<'static>;
pub type CgInputS = CgInput<'static>;
pub type ChandeInputS = ChandeInput<'static>;
pub type ChopInputS = ChopInput<'static>;
pub type CkspInputS = CkspInput<'static>;
pub type CmoInputS = CmoInput<'static>;
pub type CoppockInputS = CoppockInput<'static>;
pub type CorrelHlInputS = CorrelHlInput<'static>;
pub type CorrelationCycleInputS = CorrelationCycleInput<'static>;
pub type CviInputS = CviInput<'static>;
pub type CwmaInputS = CwmaInput<'static>;
pub type DamianiVolatmeterInputS = DamianiVolatmeterInput<'static>;
pub type DecOscInputS = DecOscInput<'static>;
pub type DecyclerInputS = DecyclerInput<'static>;
pub type DemaInputS = DemaInput<'static>;
pub type DevStopInputS = DevStopInput<'static>;
pub type DiInputS = DiInput<'static>;
pub type DmInputS = DmInput<'static>;
pub type DonchianInputS = DonchianInput<'static>;
pub type DpoInputS = DpoInput<'static>;
pub type DtiInputS = DtiInput<'static>;
pub type DxInputS = DxInput<'static>;
pub type EdcfInputS = EdcfInput<'static>;
pub type EfiInputS = EfiInput<'static>;
pub type EhlersITrendInputS = EhlersITrendInput<'static>;
pub type EmaInputS = EmaInput<'static>;
pub type EmdInputS = EmdInput<'static>;
pub type EmvInputS = EmvInput<'static>;
pub type EpmaInputS = EpmaInput<'static>;
pub type ErInputS = ErInput<'static>;
pub type EriInputS = EriInput<'static>;
pub type FisherInputS = FisherInput<'static>;
pub type FoscInputS = FoscInput<'static>;
pub type FramaInputS = FramaInput<'static>;
pub type FwmaInputS = FwmaInput<'static>;
pub type GatorOscInputS = GatorOscInput<'static>;
pub type GaussianInputS = GaussianInput<'static>;
pub type HeikinAshiInputS = HeikinAshiInput<'static>;
pub type HighPassInputS = HighPassInput<'static>;
pub type HighPass2InputS = HighPass2Input<'static>;
pub type HmaInputS = HmaInput<'static>;
pub type HtDcPeriodInputS = HtDcPeriodInput<'static>;
pub type HwmaInputS = HwmaInput<'static>;
pub type IftRsiInputS = IftRsiInput<'static>;
pub type JmaInputS = JmaInput<'static>;
pub type JsaInputS = JsaInput<'static>;
pub type KamaInputS = KamaInput<'static>;
pub type KaufmanstopInputS = KaufmanstopInput<'static>;
pub type KdjInputS = KdjInput<'static>;
pub type KeltnerInputS = KeltnerInput<'static>;
pub type KstInputS = KstInput<'static>;
pub type KurtosisInputS = KurtosisInput<'static>;
pub type KvoInputS = KvoInput<'static>;
pub type Linearreg_angleInputS = Linearreg_angleInput<'static>;
pub type LinearRegInterceptInputS = LinearRegInterceptInput<'static>;
pub type LinearRegSlopeInputS = LinearRegSlopeInput<'static>;
pub type LinRegInputS = LinRegInput<'static>;
pub type LrsiInputS = LrsiInput<'static>;
pub type MaaqInputS = MaaqInput<'static>;
pub type MabInputS = MabInput<'static>;
pub type MacdInputS = MacdInput<'static>;
pub type MamaInputS = MamaInput<'static>;
pub type MarketefiInputS = MarketefiInput<'static>;
pub type MassInputS = MassInput<'static>;
pub type MeanAdInputS = MeanAdInput<'static>;
pub type MediumAdInputS = MediumAdInput<'static>;
pub type MedpriceInputS = MedpriceInput<'static>;
pub type MfiInputS = MfiInput<'static>;
pub type MidpointInputS = MidpointInput<'static>;
pub type MidpriceInputS = MidpriceInput<'static>;
pub type MinmaxInputS = MinmaxInput<'static>;
pub type MomInputS = MomInput<'static>;
pub type MswInputS = MswInput<'static>;
pub type MwdxInputS = MwdxInput<'static>;
pub type NatrInputS = NatrInput<'static>;
pub type NmaInputS = NmaInput<'static>;
pub type PivotInputS = PivotInput<'static>;
pub type PmaInputS = PmaInput<'static>;
pub type PpoInputS = PpoInput<'static>;
pub type PviInputS = PviInput<'static>;
pub type PwmaInputS = PwmaInput<'static>;
pub type QstickInputS = QstickInput<'static>;
pub type ReflexInputS = ReflexInput<'static>;
pub type RocInputS = RocInput<'static>;
pub type RocpInputS = RocpInput<'static>;
pub type RocrInputS = RocrInput<'static>;
pub type RsiInputS = RsiInput<'static>;
pub type RviInputS = RviInput<'static>;
pub type SafeZoneStopInputS = SafeZoneStopInput<'static>;
pub type SarInputS = SarInput<'static>;
pub type SinWmaInputS = SinWmaInput<'static>;
pub type SmaInputS = SmaInput<'static>;
pub type SmmaInputS = SmmaInput<'static>;
pub type SqueezeMomentumInputS = SqueezeMomentumInput<'static>;
pub type SqwmaInputS = SqwmaInput<'static>;
pub type SrsiInputS = SrsiInput<'static>;
pub type SrwmaInputS = SrwmaInput<'static>;
pub type StcInputS = StcInput<'static>;
pub type StdDevInputS = StdDevInput<'static>;
pub type StochfInputS = StochfInput<'static>;
pub type SuperSmootherInputS = SuperSmootherInput<'static>;
pub type SuperSmoother3PoleInputS = SuperSmoother3PoleInput<'static>;
pub type SwmaInputS = SwmaInput<'static>;
pub type TemaInputS = TemaInput<'static>;
pub type TilsonInputS = TilsonInput<'static>;
pub type TrendFlexInputS = TrendFlexInput<'static>;
pub type TrimaInputS = TrimaInput<'static>;
pub type TrixInputS = TrixInput<'static>;
pub type TsiInputS = TsiInput<'static>;
pub type TtmTrendInputS = TtmTrendInput<'static>;
pub type UiInputS = UiInput<'static>;
pub type UltOscInputS = UltOscInput<'static>;
pub type VarInputS = VarInput<'static>;
pub type ViInputS = ViInput<'static>;
pub type VidyaInputS = VidyaInput<'static>;
pub type VlmaInputS = VlmaInput<'static>;
pub type VoscInputS = VoscInput<'static>;
pub type VossInputS = VossInput<'static>;
pub type VpciInputS = VpciInput<'static>;
pub type VptInputS = VptInput<'static>;
pub type VpwmaInputS = VpwmaInput<'static>;
pub type VwapInputS = VwapInput<'static>;
pub type VwmaInputS = VwmaInput<'static>;
pub type VwmacdInputS = VwmacdInput<'static>;
pub type WadInputS = WadInput<'static>;
pub type WavetrendInputS = WavetrendInput<'static>;
pub type WclpriceInputS = WclpriceInput<'static>;
pub type WildersInputS = WildersInput<'static>;
pub type WillrInputS = WillrInput<'static>;
pub type WmaInputS = WmaInput<'static>;
pub type ZlemaInputS = ZlemaInput<'static>;
pub type ZscoreInputS = ZscoreInput<'static>;

macro_rules! impl_input_len {
     ($($ty:ty),* $(,)?) => {
         $(
             impl InputLen for $ty {
                 fn with_len(len: usize) -> Self {
                     match len {
                         10_000    => Self::with_default_candles(&*CANDLES_10K),
                         100_000   => Self::with_default_candles(&*CANDLES_100K),
                         1_000_000 => Self::with_default_candles(&*CANDLES_1M),
                         _ => panic!("unsupported len {len}"),
                     }
                 }
             }
         )*
     };
 }

fn pretty_len(len: usize) -> &'static str {
    match len {
        10_000 => "10k",
        100_000 => "100k",
        1_000_000 => "1M",
        _ => panic!("unsupported len {len}"),
    }
}

const SIZES: [usize; 3] = [10_000, 100_000, 1_000_000];

fn bench_one<F, In>(
    group: &mut BenchmarkGroup<'_, criterion::measurement::WallTime>,
    label: &str,
    fun: F,
    len: usize,
    elements: Option<u64>,
) where
    F: Fn(&In) -> anyhow::Result<()> + Copy + 'static,
    In: InputLen + 'static,
{
    let input = In::with_len(len);

    if let Some(n) = elements {
        group.throughput(Throughput::Elements(n));
    }

    group.bench_with_input(
        BenchmarkId::new(label, pretty_len(len)),
        &input,
        move |b, input| b.iter(|| fun(black_box(input)).unwrap()),
    );

    group.measurement_time(Duration::from_millis(900));
    group.warm_up_time(Duration::from_millis(150));
    group.sample_size(100);
}

macro_rules! bench_scalars {
    ( $( $fun:ident => $typ:ty ),* $(,)? ) => {
        paste::paste! {
            $(
                fn [<bench_ $fun>](c: &mut Criterion) {
                    let mut group = c.benchmark_group(stringify!($fun));
                    for &len in &SIZES {
                        bench_one::<_, $typ>(&mut group, "scalar", $fun, len, None);
                    }
                    group.finish();
                }
            )*
            criterion_group!(benches_scalar, $( [<bench_ $fun>] ),*);
        }
    };
}

macro_rules! bench_variants {
    ($root:ident => $typ:ty; $elements:expr; $( $vfun:ident ),+ $(,)? ) => {
        paste::paste! {
            fn [<bench_ $root>](c: &mut Criterion) {
                let mut group = c.benchmark_group(stringify!($root));
                for &len in &SIZES {
                    $(
                        bench_one::<_, $typ>(
                            &mut group,
                            stringify!($vfun),
                            $vfun,
                            len,
                            $elements
                        );
                    )+
                }
                group.finish();
            }
            criterion_group!([<benches_ $root>], [<bench_ $root>]);
        }
    };
}

macro_rules! make_kernel_wrappers {
     ( $stem:ident, $base:path, $ityp:ty ; $( $k:ident ),+ $(,)? ) => {
         paste! {
             $(
                 #[inline(always)]
                 fn [<$stem _ $k:lower>](input: &$ityp) -> anyhow::Result<()> {
                     $base(input, Kernel::$k)
                         .map(|_| ())
                         .map_err(|e| anyhow!(e.to_string()))
                 }
             )+
         }
     };
 }

#[macro_export]
macro_rules! bench_wrappers {
      ( $( ($bench_fn:ident, $raw_fn:ident, $input_ty:ty) ),+ $(,)?) => {
          $(
              #[inline(always)]
              fn $bench_fn(input: &$input_ty) -> anyhow::Result<()> {
                  $raw_fn(input)
                      .map(|_| ())
                      .map_err(|e| anyhow::anyhow!(e.to_string()))
              }
          )+
      };
  }

macro_rules! make_batch_wrappers {
    ( $stem:ident, $builder:path, $ityp:ty ; $( $k:ident ),+ $(,)? ) => {
        paste::paste! {
            $(
                #[inline(always)]
                fn [<$stem _ $k:lower>](input: &$ityp) -> anyhow::Result<()> {
                    let slice: &[f64] = input.as_ref();
                    <$builder>::new()
                        .kernel(Kernel::$k)
                        .apply_slice(slice)?;
                    Ok(())
                }
            )+
        }
    };
}

impl_input_len!(
    AcoscInputS,
    AdInputS,
    AdoscInputS,
    AdxInputS,
    AdxrInputS,
    AlligatorInputS,
    AlmaInputS,
    AoInputS,
    ApoInputS,
    AroonInputS,
    AroonOscInputS,
    AtrInputS,
    AvgPriceInputS,
    BandPassInputS,
    BollingerBandsInputS,
    BollingerBandsWidthInputS,
    BopInputS,
    CciInputS,
    CfoInputS,
    CgInputS,
    ChandeInputS,
    ChopInputS,
    CkspInputS,
    CmoInputS,
    CoppockInputS,
    CorrelHlInputS,
    CorrelationCycleInputS,
    CviInputS,
    CwmaInputS,
    DamianiVolatmeterInputS,
    DecOscInputS,
    DecyclerInputS,
    DemaInputS,
    DevStopInputS,
    DiInputS,
    DmInputS,
    DonchianInputS,
    DpoInputS,
    DtiInputS,
    DxInputS,
    EdcfInputS,
    EfiInputS,
    EhlersITrendInputS,
    EmaInputS,
    EmdInputS,
    EmvInputS,
    EpmaInputS,
    ErInputS,
    EriInputS,
    FisherInputS,
    FoscInputS,
    FramaInputS,
    FwmaInputS,
    GatorOscInputS,
    GaussianInputS,
    HeikinAshiInputS,
    HighPassInputS,
    HighPass2InputS,
    HmaInputS,
    HtDcPeriodInputS,
    HwmaInputS,
    IftRsiInputS,
    JmaInputS,
    JsaInputS,
    KamaInputS,
    KaufmanstopInputS,
    KdjInputS,
    KeltnerInputS,
    KstInputS,
    KurtosisInputS,
    KvoInputS,
    Linearreg_angleInputS,
    LinearRegInterceptInputS,
    LinearRegSlopeInputS,
    LinRegInputS,
    LrsiInputS,
    MaaqInputS,
    MabInputS,
    MacdInputS,
    MamaInputS,
    MarketefiInputS,
    MassInputS,
    MeanAdInputS,
    MediumAdInputS,
    MedpriceInputS,
    MfiInputS,
    MidpointInputS,
    MidpriceInputS,
    MinmaxInputS,
    MomInputS,
    MswInputS,
    MwdxInputS,
    NatrInputS,
    NmaInputS,
    PivotInputS,
    PmaInputS,
    PpoInputS,
    PviInputS,
    PwmaInputS,
    QstickInputS,
    ReflexInputS,
    RocInputS,
    RocpInputS,
    RocrInputS,
    RsiInputS,
    RviInputS,
    SafeZoneStopInputS,
    SarInputS,
    SinWmaInputS,
    SmaInputS,
    SmmaInputS,
    SqueezeMomentumInputS,
    SqwmaInputS,
    SrsiInputS,
    SrwmaInputS,
    StcInputS,
    StdDevInputS,
    StochfInputS,
    SuperSmootherInputS,
    SuperSmoother3PoleInputS,
    SwmaInputS,
    TemaInputS,
    TilsonInputS,
    TrendFlexInputS,
    TrimaInputS,
    TrixInputS,
    TsiInputS,
    TtmTrendInputS,
    UiInputS,
    UltOscInputS,
    VarInputS,
    ViInputS,
    VidyaInputS,
    VlmaInputS,
    VoscInputS,
    VossInputS,
    VpciInputS,
    VptInputS,
    VpwmaInputS,
    VwapInputS,
    VwmaInputS,
    VwmacdInputS,
    WadInputS,
    WavetrendInputS,
    WclpriceInputS,
    WildersInputS,
    WillrInputS,
    WmaInputS,
    ZlemaInputS,
    ZscoreInputS
);

bench_wrappers! {
    (acosc_bench, acosc_raw, AcoscInputS),
    (ad_bench, ad_raw, AdInputS),
    (adosc_bench, adosc_raw, AdoscInputS),
    (adx_bench, adx_raw, AdxInputS),
    (adxr_bench, adxr_raw, AdxrInputS),
    (alligator_bench, alligator_raw, AlligatorInputS),
    (ao_bench, ao_raw, AoInputS),
    (apo_bench, apo_raw, ApoInputS),
    (aroon_bench, aroon_raw, AroonInputS),
    (aroon_osc_bench, aroon_osc_raw, AroonOscInputS),
    (atr_bench, atr_raw, AtrInputS),
    (avgprice_bench, avgprice_raw, AvgPriceInputS),
    (bandpass_bench, bandpass_raw, BandPassInputS),
    (bollinger_bands_bench, bollinger_bands_raw, BollingerBandsInputS),
    (bollinger_bands_width_bench, bollinger_bands_width_raw, BollingerBandsWidthInputS),
    (bop_bench, bop_raw, BopInputS),
    (cci_bench, cci_raw, CciInputS),
    (cfo_bench, cfo_raw, CfoInputS),
    (cg_bench, cg_raw, CgInputS),
    (chande_bench, chande_raw, ChandeInputS),
    (chop_bench, chop_raw, ChopInputS),
    (cksp_bench, cksp_raw, CkspInputS),
    (cmo_bench, cmo_raw, CmoInputS),
    (coppock_bench, coppock_raw, CoppockInputS),
    (correl_hl_bench, correl_hl_raw, CorrelHlInputS),
    (correlation_cycle_bench, correlation_cycle_raw, CorrelationCycleInputS),
    (cvi_bench, cvi_raw, CviInputS),
    (damiani_volatmeter_bench, damiani_volatmeter_raw, DamianiVolatmeterInputS),
    (dec_osc_bench, dec_osc_raw, DecOscInputS),
    (decycler_bench, decycler_raw, DecyclerInputS),
    (devstop_bench, devstop_raw, DevStopInputS),
    (di_bench, di_raw, DiInputS),
    (dm_bench, dm_raw, DmInputS),
    (donchian_bench, donchian_raw, DonchianInputS),
    (dpo_bench, dpo_raw, DpoInputS),
    (dti_bench, dti_raw, DtiInputS),
    (dx_bench, dx_raw, DxInputS),
    (efi_bench, efi_raw, EfiInputS),
    (emd_bench, emd_raw, EmdInputS),
    (emv_bench, emv_raw, EmvInputS),
    (er_bench, er_raw, ErInputS),
    (eri_bench, eri_raw, EriInputS),
    (fisher_bench, fisher_raw, FisherInputS),
    (fosc_bench, fosc_raw, FoscInputS),
    (gatorosc_bench, gatorosc_raw, GatorOscInputS),
    (gaussian_bench, gaussian_raw, GaussianInputS),
    (heikin_ashi_candles_bench, heikin_ashi_candles_raw, HeikinAshiInputS),
    (highpass_bench, highpass_raw, HighPassInputS),
    (highpass_2_pole_bench, highpass_2_pole_raw, HighPass2InputS),
    (hma_bench, hma_raw, HmaInputS),
    (ht_dcperiod_bench, ht_dcperiod_raw, HtDcPeriodInputS),
    (hwma_bench, hwma_raw, HwmaInputS),
    (ift_rsi_bench, ift_rsi_raw, IftRsiInputS),
    (jma_bench, jma_raw, JmaInputS),
    (jsa_bench, jsa_raw, JsaInputS),
    (kama_bench, kama_raw, KamaInputS),
    (kaufmanstop_bench, kaufmanstop_raw, KaufmanstopInputS),
    (kdj_bench, kdj_raw, KdjInputS),
    (keltner_bench, keltner_raw, KeltnerInputS),
    (kst_bench, kst_raw, KstInputS),
    (kurtosis_bench, kurtosis_raw, KurtosisInputS),
    (kvo_bench, kvo_raw, KvoInputS),
    (linearreg_angle_bench, linearreg_angle_raw, Linearreg_angleInputS),
    (linearreg_intercept_bench, linearreg_intercept_raw, LinearRegInterceptInputS),
    (linearreg_slope_bench, linearreg_slope_raw, LinearRegSlopeInputS),
    (linreg_bench, linreg_raw, LinRegInputS),
    (lrsi_bench, lrsi_raw, LrsiInputS),
    (maaq_bench, maaq_raw, MaaqInputS),
    (mab_bench, mab_raw, MabInputS),
    (macd_bench, macd_raw, MacdInputS),
    (mama_bench, mama_raw, MamaInputS),
    (marketfi_bench, marketfi_raw, MarketefiInputS),
    (mass_bench, mass_raw, MassInputS),
    (mean_ad_bench, mean_ad_raw, MeanAdInputS),
    (medium_ad_bench, medium_ad_raw, MediumAdInputS),
    (medprice_bench, medprice_raw, MedpriceInputS),
    (mfi_bench, mfi_raw, MfiInputS),
    (midpoint_bench, midpoint_raw, MidpointInputS),
    (midprice_bench, midprice_raw, MidpriceInputS),
    (minmax_bench, minmax_raw, MinmaxInputS),
    (mom_bench, mom_raw, MomInputS),
    (msw_bench, msw_raw, MswInputS),
    (mwdx_bench, mwdx_raw, MwdxInputS),
    (natr_bench, natr_raw, NatrInputS),
    (nma_bench, nma_raw, NmaInputS),
    (pivot_bench, pivot_raw, PivotInputS),
    (pma_bench, pma_raw, PmaInputS),
    (ppo_bench, ppo_raw, PpoInputS),
    (pvi_bench, pvi_raw, PviInputS),
    (pwma_bench, pwma_raw, PwmaInputS),
    (qstick_bench, qstick_raw, QstickInputS),
    (reflex_bench, reflex_raw, ReflexInputS),
    (roc_bench, roc_raw, RocInputS),
    (rocp_bench, rocp_raw, RocpInputS),
    (rocr_bench, rocr_raw, RocrInputS),
    (rsi_bench, rsi_raw, RsiInputS),
    (rvi_bench, rvi_raw, RviInputS),
    (safezonestop_bench, safezonestop_raw, SafeZoneStopInputS),
    (sar_bench, sar_raw, SarInputS),
    (sinwma_bench, sinwma_raw, SinWmaInputS),
    (sma_bench, sma_raw, SmaInputS),
    (smma_bench, smma_raw, SmmaInputS),
    (squeeze_momentum_bench, squeeze_momentum_raw, SqueezeMomentumInputS),
    (sqwma_bench, sqwma_raw, SqwmaInputS),
    (srsi_bench, srsi_raw, SrsiInputS),
    (srwma_bench, srwma_raw, SrwmaInputS),
    (stc_bench, stc_raw, StcInputS),
    (stddev_bench, stddev_raw, StdDevInputS),
    (stochf_bench, stochf_raw, StochfInputS),
    (supersmoother_bench, supersmoother_raw, SuperSmootherInputS),
    (supersmoother_3_pole_bench, supersmoother_3_pole_raw, SuperSmoother3PoleInputS),
    (swma_bench, swma_raw, SwmaInputS),
    (tema_bench, tema_raw, TemaInputS),
    (tilson_bench, tilson_raw, TilsonInputS),
    (trendflex_bench, trendflex_raw, TrendFlexInputS),
    (trima_bench, trima_raw, TrimaInputS),
    (trix_bench, trix_raw, TrixInputS),
    (tsi_bench, tsi_raw, TsiInputS),
    (ttm_trend_bench, ttm_trend_raw, TtmTrendInputS),
    (ui_bench, ui_raw, UiInputS),
    (ultosc_bench, ultosc_raw, UltOscInputS),
    (var_bench, var_raw, VarInputS),
    (vi_bench, vi_raw, ViInputS),
    (vidya_bench, vidya_raw, VidyaInputS),
    (vlma_bench, vlma_raw, VlmaInputS),
    (vosc_bench, vosc_raw, VoscInputS),
    (voss_bench, voss_raw, VossInputS),
    (vpci_bench, vpci_raw, VpciInputS),
    (vpt_bench, vpt_raw, VptInputS),
    (vpwma_bench, vpwma_raw, VpwmaInputS),
    (vwap_bench, vwap_raw, VwapInputS),
    (vwma_bench, vwma_raw, VwmaInputS),
    (vwmacd_bench, vwmacd_raw, VwmacdInputS),
    (wad_bench, wad_raw, WadInputS),
    (wavetrend_bench, wavetrend_raw, WavetrendInputS),
    (wclprice_bench, wclprice_raw, WclpriceInputS),
    (wilders_bench, wilders_raw, WildersInputS),
    (willr_bench, willr_raw, WillrInputS),
    (wma_bench, wma_raw, WmaInputS),
    (zlema_bench, zlema_raw, ZlemaInputS),
    (zscore_bench, zscore_raw, ZscoreInputS),
}

bench_scalars!(
    acosc_bench => AcoscInputS,
    ad_bench    => AdInputS,
    adosc_bench => AdoscInputS,
    adx_bench   => AdxInputS,
    adxr_bench  => AdxrInputS,
    alligator_bench => AlligatorInputS,

    ao_bench   => AoInputS,
    apo_bench  => ApoInputS,

    aroon_bench        => AroonInputS,
    aroon_osc_bench    => AroonOscInputS,
    atr_bench          => AtrInputS,
    avgprice_bench     => AvgPriceInputS,
    bandpass_bench     => BandPassInputS,

    bollinger_bands_bench => BollingerBandsInputS,
    bollinger_bands_width_bench => BollingerBandsWidthInputS,
    bop_bench         => BopInputS,
    cci_bench         => CciInputS,
    cfo_bench         => CfoInputS,
    cg_bench          => CgInputS,
    chande_bench      => ChandeInputS,
    chop_bench        => ChopInputS,
    cksp_bench        => CkspInputS,
    cmo_bench         => CmoInputS,
    coppock_bench     => CoppockInputS,
    correl_hl_bench   => CorrelHlInputS,
    correlation_cycle_bench => CorrelationCycleInputS,
    cvi_bench         => CviInputS,
    damiani_volatmeter_bench => DamianiVolatmeterInputS,
    dec_osc_bench     => DecOscInputS,
    decycler_bench    => DecyclerInputS,
    devstop_bench     => DevStopInputS,
    di_bench          => DiInputS,
    dm_bench          => DmInputS,
    donchian_bench    => DonchianInputS,
    dpo_bench         => DpoInputS,
    dti_bench         => DtiInputS,
    dx_bench          => DxInputS,
    efi_bench         => EfiInputS,
    emd_bench         => EmdInputS,
    emv_bench         => EmvInputS,
    er_bench          => ErInputS,
    eri_bench         => EriInputS,
    fisher_bench      => FisherInputS,
    fosc_bench        => FoscInputS,
    gatorosc_bench    => GatorOscInputS,
    gaussian_bench    => GaussianInputS,
    heikin_ashi_candles_bench => HeikinAshiInputS,
    highpass_bench    => HighPassInputS,
    highpass_2_pole_bench => HighPass2InputS,
    hma_bench         => HmaInputS,
    ht_dcperiod_bench => HtDcPeriodInputS,
    hwma_bench        => HwmaInputS,
    ift_rsi_bench     => IftRsiInputS,
    jma_bench         => JmaInputS,
    jsa_bench         => JsaInputS,
    kama_bench        => KamaInputS,
    kaufmanstop_bench => KaufmanstopInputS,
    kdj_bench         => KdjInputS,
    keltner_bench     => KeltnerInputS,
    kst_bench         => KstInputS,
    kurtosis_bench    => KurtosisInputS,
    kvo_bench         => KvoInputS,

    linearreg_angle_bench     => Linearreg_angleInputS,
    linearreg_intercept_bench => LinearRegInterceptInputS,
    linearreg_slope_bench     => LinearRegSlopeInputS,
    linreg_bench              => LinRegInputS,
    lrsi_bench                => LrsiInputS,

    maaq_bench => MaaqInputS,
    mab_bench  => MabInputS,
    macd_bench => MacdInputS,
    mama_bench => MamaInputS,
    marketfi_bench  => MarketefiInputS,
    mass_bench      => MassInputS,
    mean_ad_bench   => MeanAdInputS,
    medium_ad_bench => MediumAdInputS,
    medprice_bench  => MedpriceInputS,
    mfi_bench       => MfiInputS,
    midpoint_bench  => MidpointInputS,
    midprice_bench  => MidpriceInputS,
    minmax_bench    => MinmaxInputS,
    mom_bench       => MomInputS,
    msw_bench       => MswInputS,
    mwdx_bench      => MwdxInputS,

    natr_bench   => NatrInputS,
    nma_bench    => NmaInputS,
    pivot_bench  => PivotInputS,
    pma_bench    => PmaInputS,
    ppo_bench    => PpoInputS,
    pvi_bench    => PviInputS,
    pwma_bench   => PwmaInputS,
    qstick_bench => QstickInputS,
    reflex_bench => ReflexInputS,
    roc_bench    => RocInputS,
    rocp_bench   => RocpInputS,
    rocr_bench   => RocrInputS,
    rsi_bench    => RsiInputS,
    rvi_bench    => RviInputS,
    safezonestop_bench => SafeZoneStopInputS,
    sar_bench    => SarInputS,
    sinwma_bench => SinWmaInputS,
    sma_bench    => SmaInputS,
    smma_bench   => SmmaInputS,
    squeeze_momentum_bench => SqueezeMomentumInputS,
    sqwma_bench  => SqwmaInputS,
    srsi_bench   => SrsiInputS,
    srwma_bench  => SrwmaInputS,
    stc_bench    => StcInputS,
    stddev_bench => StdDevInputS,
    stochf_bench => StochfInputS,
    supersmoother_bench => SuperSmootherInputS,
    supersmoother_3_pole_bench => SuperSmoother3PoleInputS,
    swma_bench   => SwmaInputS,
    tema_bench   => TemaInputS,
    tilson_bench => TilsonInputS,
    trendflex_bench => TrendFlexInputS,
    trima_bench  => TrimaInputS,
    trix_bench   => TrixInputS,
    tsi_bench    => TsiInputS,
    ttm_trend_bench => TtmTrendInputS,
    ui_bench     => UiInputS,
    ultosc_bench => UltOscInputS,
    var_bench    => VarInputS,
    vi_bench     => ViInputS,
    vidya_bench  => VidyaInputS,
    vlma_bench   => VlmaInputS,
    vosc_bench   => VoscInputS,
    voss_bench   => VossInputS,
    vpci_bench   => VpciInputS,
    vpt_bench    => VptInputS,
    vpwma_bench  => VpwmaInputS,
    vwap_bench   => VwapInputS,
    vwma_bench   => VwmaInputS,
    vwmacd_bench => VwmacdInputS,
    wad_bench    => WadInputS,
    wavetrend_bench => WavetrendInputS,
    wclprice_bench => WclpriceInputS,
    wilders_bench   => WildersInputS,
    willr_bench     => WillrInputS,
    wma_bench       => WmaInputS,
    zlema_bench     => ZlemaInputS,
    zscore_bench    => ZscoreInputS
);

make_kernel_wrappers!(alma, alma_with_kernel, AlmaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(cwma, cwma_with_kernel, CwmaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(dema, dema_with_kernel, DemaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(edcf, edcf_with_kernel, EdcfInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(ehlers_itrend, ehlers_itrend_with_kernel, EhlersITrendInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(ema, ema_with_kernel, EmaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(epma, epma_with_kernel, EpmaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(frama, frama_with_kernel, FramaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(fwma, fwma_with_kernel, FwmaInputS; Scalar,Avx2,Avx512);

make_batch_wrappers!(
    alma_batch, AlmaBatchBuilder, AlmaInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

make_batch_wrappers!(
    cwma_batch, CwmaBatchBuilder, CwmaInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

make_batch_wrappers!(
    dema_batch, DemaBatchBuilder, DemaInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

make_batch_wrappers!(
    edcf_batch, EdcfBatchBuilder, EdcfInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

make_batch_wrappers!(
    ehlers_itrend_batch, EhlersITrendBatchBuilder, EhlersITrendInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

make_batch_wrappers!(
    ema_batch, EmaBatchBuilder, EmaInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

make_batch_wrappers!(
    epma_batch, EpmaBatchBuilder, EpmaInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

make_batch_wrappers!(
    frama_batch, FramaBatchBuilder, FramaInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

make_batch_wrappers!(
    fwma_batch, FwmaBatchBuilder, FwmaInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

bench_variants!(
    alma_batch => AlmaInputS; Some(232);
    alma_batch_scalarbatch,
    alma_batch_avx2batch,
    alma_batch_avx512batch,
);

bench_variants!(
    cwma_batch => CwmaInputS; Some(227);
    cwma_batch_scalarbatch,
    cwma_batch_avx2batch,
    cwma_batch_avx512batch,
);

bench_variants!(
   dema_batch => DemaInputS; Some(227);
   dema_batch_scalarbatch,
   dema_batch_avx2batch,
   dema_batch_avx512batch,
);

bench_variants!(
   edcf_batch => EdcfInputS; Some(227);
   edcf_batch_scalarbatch,
   edcf_batch_avx2batch,
   edcf_batch_avx512batch,
);

bench_variants!(
    ehlers_itrend_batch => EhlersITrendInputS; Some(227);
    ehlers_itrend_batch_scalarbatch,
    ehlers_itrend_batch_avx2batch,
    ehlers_itrend_batch_avx512batch,
);

bench_variants!(
    ema_batch => EmaInputS; Some(227);
    ema_batch_scalarbatch,
    ema_batch_avx2batch,
    ema_batch_avx512batch,
);

bench_variants!(
    epma_batch => EpmaInputS; Some(227);
    epma_batch_scalarbatch,
    epma_batch_avx2batch,
    epma_batch_avx512batch,
);

bench_variants!(
    frama_batch => FramaInputS; Some(227);
    frama_batch_scalarbatch,
    frama_batch_avx2batch,
    frama_batch_avx512batch,
);

bench_variants!(
    fwma_batch => FwmaInputS; Some(227);
    fwma_batch_scalarbatch,
    fwma_batch_avx2batch,
    fwma_batch_avx512batch,
);

bench_variants!(
    alma => AlmaInputS; None;
    alma_scalar,
    alma_avx2,
    alma_avx512,
);

bench_variants!(
   cwma => CwmaInputS; None;
   cwma_scalar,
   cwma_avx2,
   cwma_avx512,
);

bench_variants!(
   dema => DemaInputS; None;
   dema_scalar,
   dema_avx2,
   dema_avx512,
);

bench_variants!(
    edcf => EdcfInputS; None;
    edcf_scalar,
    edcf_avx2,
    edcf_avx512,
);

bench_variants!(
    ehlers_itrend => EhlersITrendInputS; None;
    ehlers_itrend_scalar,
    ehlers_itrend_avx2,
    ehlers_itrend_avx512,
);

bench_variants!(
    ema => EmaInputS; None;
    ema_scalar,
    ema_avx2,
    ema_avx512,
);

bench_variants!(
    epma => EpmaInputS; None;
    epma_scalar,
    epma_avx2,
    epma_avx512,
);

bench_variants!(
    frama => FramaInputS; None;
    frama_scalar,
    frama_avx2,
    frama_avx512,
);

bench_variants!(
    fwma => FwmaInputS; None;
    fwma_scalar,
    fwma_avx2,
    fwma_avx512,
);

criterion_main!(
    benches_scalar,
    benches_alma,
    benches_alma_batch,
    benches_cwma,
    benches_cwma_batch,
    benches_dema,
    benches_dema_batch,
    benches_edcf,
    benches_edcf_batch,
    benches_ehlers_itrend,
    benches_ehlers_itrend_batch,
    benches_ema,
    benches_ema_batch,
    benches_epma,
    benches_epma_batch,
    benches_frama,
    benches_frama_batch,
    benches_fwma,
    benches_fwma_batch
);
