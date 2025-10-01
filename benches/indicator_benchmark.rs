// NOTE: Bench infra
// - For OHLC-based indicators (e.g., SuperTrend, WILLR), the batch macros here
//   currently assume single-slice inputs. We should add a dedicated OHLC bench
//   wrapper macro (e.g., `make_ohlc_batch_wrappers!`) that takes (high, low, close)
//   to avoid one-off custom wrappers.
// - Until that macro exists, avoid adding per-indicator custom bench code.

use anyhow::anyhow;
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion, Throughput,
};
use my_project::utilities::enums::Kernel;
use once_cell::sync::Lazy;
use paste::paste;
use std::time::Duration;

// Import moving averages separately
use my_project::indicators::moving_averages::{
    alma::{alma_with_kernel, AlmaBatchBuilder, AlmaInput},
    buff_averages::{
        buff_averages, buff_averages_with_kernel, BuffAveragesBatchBuilder, BuffAveragesInput,
    },
    cwma::{cwma_with_kernel, CwmaBatchBuilder, CwmaInput},
    dema::{dema_with_kernel, DemaBatchBuilder, DemaInput},
    dma::{dma_with_kernel, DmaBatchBuilder, DmaInput},
    edcf::{edcf_with_kernel, EdcfBatchBuilder, EdcfInput},
    ehlers_ecema::{ehlers_ecema_with_kernel, EhlersEcemaBatchBuilder, EhlersEcemaInput},
    ehlers_itrend::{ehlers_itrend_with_kernel, EhlersITrendBatchBuilder, EhlersITrendInput},
    ehlers_kama::{ehlers_kama_with_kernel, EhlersKamaBatchBuilder, EhlersKamaInput},
    ehlers_pma::{ehlers_pma_with_kernel, EhlersPmaBuilder, EhlersPmaInput},
    ehma::{ehma_with_kernel, EhmaBatchBuilder, EhmaInput},
    ema::{ema_with_kernel, EmaBatchBuilder, EmaInput},
    epma::{epma_with_kernel, EpmaBatchBuilder, EpmaInput},
    frama::{frama_with_kernel, FramaBatchBuilder, FramaInput},
    fwma::{fwma_with_kernel, FwmaBatchBuilder, FwmaInput},
    gaussian::{gaussian_with_kernel, GaussianBatchBuilder, GaussianInput},
    highpass::{highpass_with_kernel, HighPassBatchBuilder, HighPassInput},
    highpass_2_pole::{highpass_2_pole_with_kernel, HighPass2BatchBuilder, HighPass2Input},
    hma::{hma_with_kernel, HmaBatchBuilder, HmaInput},
    hwma::{hwma_with_kernel, HwmaBatchBuilder, HwmaInput},
    jma::{jma_with_kernel, JmaBatchBuilder, JmaInput},
    jsa::{jsa_with_kernel, JsaBatchBuilder, JsaInput},
    kama::{kama_with_kernel, KamaBatchBuilder, KamaInput},
    linreg::{linreg_with_kernel, LinRegBatchBuilder, LinRegInput},
    maaq::{maaq_with_kernel, MaaqBatchBuilder, MaaqInput},
    mama::{mama_with_kernel, MamaBatchBuilder, MamaInput},
    mwdx::{mwdx_with_kernel, MwdxBatchBuilder, MwdxInput},
    nama::{nama_with_kernel, NamaBatchBuilder, NamaInput},
    nma::{nma_with_kernel, NmaBatchBuilder, NmaInput},
    pwma::{pwma_with_kernel, PwmaBatchBuilder, PwmaInput},
    reflex::{reflex_with_kernel, ReflexBatchBuilder, ReflexInput},
    sama::{sama_with_kernel, SamaBatchBuilder, SamaInput},
    sinwma::{sinwma_with_kernel, SinWmaBatchBuilder, SinWmaInput},
    sma::{sma_with_kernel, SmaBatchBuilder, SmaInput},
    smma::{smma_with_kernel, SmmaBatchBuilder, SmmaInput},
    sqwma::{sqwma_with_kernel, SqwmaBatchBuilder, SqwmaInput},
    srwma::{srwma_with_kernel, SrwmaBatchBuilder, SrwmaInput},
    supersmoother::{supersmoother_with_kernel, SuperSmootherBatchBuilder, SuperSmootherInput},
    supersmoother_3_pole::{
        supersmoother_3_pole_with_kernel, SuperSmoother3PoleBatchBuilder, SuperSmoother3PoleInput,
    },
    swma::{swma_with_kernel, SwmaBatchBuilder, SwmaInput},
    tema::{tema_with_kernel, TemaBatchBuilder, TemaInput},
    tilson::{tilson_with_kernel, TilsonBatchBuilder, TilsonInput},
    tradjema::{tradjema_with_kernel, TradjemaInput},
    trendflex::{trendflex_with_kernel, TrendFlexBatchBuilder, TrendFlexInput},
    trima::{trima_with_kernel, TrimaBatchBuilder, TrimaInput},
    uma::{uma_with_kernel, UmaBatchBuilder, UmaInput},
    volume_adjusted_ma::{
        VolumeAdjustedMa, VolumeAdjustedMaBatchBuilder, VolumeAdjustedMaInput,
        VolumeAdjustedMa_with_kernel,
    },
    vpwma::{vpwma_with_kernel, VpwmaBatchBuilder, VpwmaInput},
    vwma::{vwma_with_kernel, VwmaInput, VwmaParams},
    wilders::{wilders_with_kernel, WildersBatchBuilder, WildersInput},
    wma::{wma_with_kernel, WmaBatchBuilder, WmaInput},
    zlema::{zlema_with_kernel, ZlemaBatchBuilder, ZlemaInput},
};

use my_project::indicators::{
    bandpass::{bandpass as bandpass_raw, bandpass_with_kernel, BandPassBatchBuilder, BandPassInput},
    cci_cycle::{cci_cycle, cci_cycle_with_kernel, CciCycleBatchBuilder, CciCycleInput},
    fvg_trailing_stop::{fvg_trailing_stop, fvg_trailing_stop_with_kernel, FvgTrailingStopInput},
    halftrend::{halftrend, halftrend_with_kernel, HalfTrendInput},
    net_myrsi::{net_myrsi, net_myrsi_with_kernel, NetMyrsiBatchBuilder, NetMyrsiInput},
    reverse_rsi::{reverse_rsi, reverse_rsi_with_kernel, ReverseRsiBatchBuilder, ReverseRsiInput},
};

use my_project::indicators::moving_averages::volatility_adjusted_ma::{
    vama, vama_with_kernel, VamaBatchBuilder, VamaInput as VamaInputMv,
};

// Removed - other_indicators no longer exists

use my_project::indicators::{
    acosc::{acosc as acosc_raw, AcoscInput},
    ad::{ad as ad_raw, ad_with_kernel, AdInput},
    adosc::{adosc as adosc_raw, AdoscInput},
    adx::{adx as adx_raw, adx_with_kernel, AdxBatchBuilder, AdxInput},
    adxr::{adxr as adxr_raw, adxr_with_kernel, AdxrInput},
    alligator::{alligator as alligator_raw, AlligatorInput},
    alphatrend::{alphatrend as alphatrend_raw, alphatrend_with_kernel, AlphaTrendInput},
    ao::{ao as ao_raw, AoInput},
    apo::{apo as apo_raw, apo_with_kernel, ApoBatchBuilder, ApoInput},
    aroon::{aroon as aroon_raw, AroonInput},
    aroonosc::{aroon_osc as aroon_osc_raw, AroonOscInput},
    atr::{atr as atr_raw, AtrInput},
    avsl::{avsl_with_kernel, AvslBatchBuilder, AvslInput},
    // bandpass import moved above to group with kernel builder/use
    bollinger_bands::{
        bollinger_bands as bollinger_bands_raw, bollinger_bands_with_kernel,
        BollingerBandsBatchBuilder, BollingerBandsInput,
    },
    bollinger_bands_width::{
        bollinger_bands_width as bollinger_bands_width_raw,
        bollinger_bands_width_with_kernel,
        BollingerBandsWidthBatchBuilder,
        BollingerBandsWidthInput,
    },
    bop::{bop as bop_raw, BopInput},
    cci::{cci as cci_raw, cci_with_kernel, CciInput},
    cfo::{cfo as cfo_raw, cfo_with_kernel, CfoBatchBuilder, CfoInput},
    cg::{cg as cg_raw, cg_with_kernel, CgInput},
    chande::{chande as chande_raw, ChandeInput},
    chandelier_exit::{chandelier_exit_with_kernel, CeBatchBuilder, ChandelierExitInput},
    chop::{chop as chop_raw, ChopInput},
    cksp::{cksp as cksp_raw, cksp_with_kernel, CkspBatchBuilder, CkspInput},
    cmo::{cmo as cmo_raw, cmo_with_kernel, CmoBatchBuilder, CmoInput},
    coppock::{coppock as coppock_raw, CoppockInput},
    cora_wave::{cora_wave as cora_wave_raw, CoraWaveBatchBuilder, CoraWaveInput},
    correl_hl::{correl_hl as correl_hl_raw, CorrelHlInput},
    correlation_cycle::{
        correlation_cycle as correlation_cycle_raw, correlation_cycle_with_kernel,
        CorrelationCycleBatchBuilder, CorrelationCycleInput,
    },
    cvi::{cvi as cvi_raw, cvi_with_kernel, CviBatchBuilder, CviInput},
    damiani_volatmeter::{damiani_volatmeter as damiani_volatmeter_raw, DamianiVolatmeterInput},
    dec_osc::{dec_osc as dec_osc_raw, DecOscInput},
    decycler::{decycler as decycler_raw, DecyclerInput},
    devstop::{devstop as devstop_raw, DevStopInput},
    di::{di as di_raw, DiInput},
    dm::{dm as dm_raw, DmBatchBuilder, DmData, DmInput},
    donchian::{donchian as donchian_raw, donchian_with_kernel, DonchianInput},
    dpo::{dpo as dpo_raw, dpo_with_kernel, DpoBatchBuilder, DpoInput},
    dti::{dti as dti_raw, dti_with_kernel, DtiInput},
    dx::{dx as dx_raw, DxBatchBuilder, DxData, DxInput},
    efi::{efi as efi_raw, efi_with_kernel, EfiInput},
    emd::{emd as emd_raw, EmdInput},
    emv::{emv as emv_raw, emv_with_kernel, EmvInput},
    er::{er as er_raw, er_with_kernel, ErBatchBuilder, ErInput},
    eri::{eri as eri_raw, eri_with_kernel, EriInput},
    fisher::{fisher as fisher_raw, FisherInput},
    fosc::{fosc as fosc_raw, fosc_with_kernel, FoscInput},
    gatorosc::{gatorosc as gatorosc_raw, GatorOscInput},
    ift_rsi::{ift_rsi as ift_rsi_raw, IftRsiInput},
    kaufmanstop::{kaufmanstop as kaufmanstop_raw, kaufmanstop_with_kernel, KaufmanstopInput},
    kdj::{kdj as kdj_raw, KdjInput},
    keltner::{keltner as keltner_raw, KeltnerInput},
    kst::{kst as kst_raw, KstBatchBuilder, KstInput},
    kurtosis::{kurtosis as kurtosis_raw, kurtosis_with_kernel, KurtosisInput},
    kvo::{kvo as kvo_raw, KvoBatchBuilder, KvoInput},
    linearreg_angle::{
        linearreg_angle as linearreg_angle_raw, linearreg_angle_with_kernel, Linearreg_angleInput,
    },
    linearreg_intercept::{
        linearreg_intercept as linearreg_intercept_raw, LinearRegInterceptInput,
    },
    linearreg_slope::{
        linearreg_slope as linearreg_slope_raw, linearreg_slope_with_kernel, LinearRegSlopeBatchBuilder,
        LinearRegSlopeInput,
    },
    lpc::{lpc as lpc_raw, LpcInput},
    lrsi::{lrsi as lrsi_raw, LrsiInput},
    mab::{mab as mab_raw, mab_with_kernel, MabInput},
    macd::{macd as macd_raw, MacdInput},
    macz::{macz_with_kernel, MaczBatchBuilder, MaczInput},
    marketefi::{marketefi as marketfi_raw, marketefi_with_kernel, MarketefiInput},
    mass::{mass as mass_raw, MassInput},
    mean_ad::{mean_ad as mean_ad_raw, MeanAdInput},
    medium_ad::{medium_ad as medium_ad_raw, MediumAdInput},
    medprice::{medprice as medprice_raw, MedpriceInput},
    mfi::{mfi as mfi_raw, MfiBatchBuilder, MfiBatchRange, MfiData, MfiInput},
    midpoint::{midpoint as midpoint_raw, MidpointInput},
    midprice::{midprice as midprice_raw, MidpriceInput},
    minmax::{minmax as minmax_raw, MinmaxInput},
    mod_god_mode::{mod_god_mode as mod_god_mode_raw, mod_god_mode_with_kernel, ModGodModeInput},
    mom::{mom as mom_raw, mom_with_kernel, MomBatchBuilder, MomInput},
    msw::{msw as msw_raw, MswInput},
    nadaraya_watson_envelope::{
        nadaraya_watson_envelope as nadaraya_watson_envelope_raw,
        nadaraya_watson_envelope_with_kernel,
        NweInput,
    },
    natr::{natr as natr_raw, natr_with_kernel, NatrBatchBuilder, NatrData, NatrInput},
    nvi::{nvi as nvi_raw, nvi_with_kernel, NviInput},
    obv::{obv as obv_raw, ObvInput},
    ott::{ott as ott_raw, ott_with_kernel, OttBatchBuilder, OttInput},
    otto::{otto as otto_raw, OttoBatchBuilder, OttoInput},
    percentile_nearest_rank::{
        percentile_nearest_rank_with_kernel, PercentileNearestRankBatchBuilder,
        PercentileNearestRankInput,
    },
    pfe::{pfe as pfe_raw, PfeBatchBuilder, PfeInput},
    pivot::{pivot as pivot_raw, pivot_with_kernel, PivotBatchBuilder, PivotData, PivotInput},
    pma::{pma as pma_raw, pma_with_kernel, PmaBatchBuilder, PmaInput},
    ppo::{ppo as ppo_raw, ppo_with_kernel, PpoInput},
    prb::{prb as prb_raw, PrbBatchBuilder, PrbInput},
    pvi::{pvi as pvi_raw, pvi_with_kernel, PviBatchBuilder, PviInput},
    qqe::{qqe as qqe_raw, QqeInput},
    qstick::{qstick as qstick_raw, qstick_with_kernel, QstickBatchBuilder, QstickBatchRange, QstickData, QstickInput},
    range_filter::{range_filter_with_kernel, RangeFilterBatchBuilder, RangeFilterInput},
    roc::{roc as roc_raw, RocInput},
    rocp::{rocp as rocp_raw, RocpInput},
    rocr::{rocr as rocr_raw, rocr_with_kernel, RocrInput},
    rsi::{rsi as rsi_raw, RsiInput},
    rsmk::{rsmk as rsmk_raw, RsmkInput},
    rsx::{rsx as rsx_raw, rsx_with_kernel, RsxBatchBuilder, RsxInput},
    rvi::{rvi as rvi_raw, RviInput},
    safezonestop::{safezonestop as safezonestop_raw, SafeZoneStopInput},
    sar::{sar as sar_raw, sar_with_kernel, SarBatchBuilder, SarData, SarInput},
    squeeze_momentum::{
        squeeze_momentum as squeeze_momentum_raw, SqueezeMomentumBatchBuilder, SqueezeMomentumInput,
    },
    srsi::{srsi as srsi_raw, srsi_with_kernel, SrsiBatchBuilder, SrsiInput},
    stc::{stc as stc_raw, StcInput},
    stddev::{stddev as stddev_raw, StdDevBatchBuilder, StdDevInput},
    stoch::{stoch as stoch_raw, stoch_with_kernel, StochBatchBuilder, StochData, StochInput},
    stochf::{stochf as stochf_raw, stochf_with_kernel, StochfInput},
    supertrend::{supertrend as supertrend_raw, supertrend_with_kernel, SuperTrendInput},
    trix::{trix_with_kernel, TrixBatchBuilder, TrixInput},
    tsf::{tsf as tsf_raw, tsf_with_kernel, TsfInput},
    tsi::{tsi as tsi_raw, TsiInput},
    ttm_squeeze::{ttm_squeeze as ttm_squeeze_raw, TtmSqueezeInput},
    ttm_trend::{ttm_trend as ttm_trend_raw, TtmTrendInput},
    ui::{ui as ui_raw, ui_with_kernel, UiInput},
    ultosc::{
        ultosc as ultosc_raw, ultosc_with_kernel, UltOscBatchBuilder, UltOscBatchRange, UltOscData,
        UltOscInput,
    },
    var::{var as var_raw, var_with_kernel, VarBatchBuilder, VarInput},
    vi::{vi as vi_raw, ViBatchBuilder, ViData, ViInput},
    vidya::{vidya_with_kernel, VidyaBatchBuilder, VidyaInput},
    vlma::{vlma_with_kernel, VlmaBatchBuilder, VlmaInput},
    vosc::{vosc as vosc_raw, VoscInput, VoscBatchBuilder},
    voss::{voss as voss_raw, VossInput},
    vpci::{vpci as vpci_raw, VpciInput},
    vpt::{vpt as vpt_raw, VptInput},
    vwap::{vwap as vwap_raw, VwapInput},
    vwmacd::{vwmacd as vwmacd_raw, VwmacdInput},
    wad::{wad as wad_raw, WadInput},
    wavetrend::{
        wavetrend as wavetrend_raw, wavetrend_with_kernel, WavetrendBatchBuilder, WavetrendInput,
    },
    wclprice::{wclprice as wclprice_raw, wclprice_with_kernel, WclpriceInput},
    willr::{willr as willr_raw, WillrBatchBuilder, WillrBuilder, WillrData, WillrInput},
    wto::{wto_with_kernel, WtoBatchBuilder, WtoInput},
    zscore::{zscore as zscore_raw, zscore_with_kernel, ZscoreBatchBuilder, ZscoreInput},
    deviation::{deviation as deviation_raw, deviation_with_kernel, DeviationBatchBuilder, DeviationInput},
    // ASO (Average Sentiment Oscillator)
    aso::{aso_with_kernel, AsoBatchBuilder, AsoData, AsoInput},
};
use my_project::indicators::correl_hl::correl_hl_with_kernel;
// DVDIQQE indicator (single-series)
use my_project::indicators::dvdiqqe::{dvdiqqe_with_kernel, DvdiqqeInput};

// Bring stc_with_kernel for kernel-specific benches
use my_project::indicators::stc::stc_with_kernel;
// DM kernel entry for single-series variant benches
use my_project::indicators::dm::dm_with_kernel;

use my_project::utilities::data_loader::{read_candles_from_csv, source_type, Candles};

// (other_indicators section removed - NAMA moved to moving_averages)

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
pub type AlphaTrendInputS = AlphaTrendInput<'static>;
pub type AlmaInputS = AlmaInput<'static>;
pub type MaczInputS = MaczInput<'static>;
pub type AoInputS = AoInput<'static>;
pub type ApoInputS = ApoInput<'static>;
pub type AroonInputS = AroonInput<'static>;
pub type AroonOscInputS = AroonOscInput<'static>;
pub type AtrInputS = AtrInput<'static>;
pub type BandPassInputS = BandPassInput<'static>;
pub type BollingerBandsInputS = BollingerBandsInput<'static>;
pub type BollingerBandsWidthInputS = BollingerBandsWidthInput<'static>;
pub type BopInputS = BopInput<'static>;
pub type BuffAveragesInputS = BuffAveragesInput<'static>;
pub type CciInputS = CciInput<'static>;
pub type CfoInputS = CfoInput<'static>;
pub type CgInputS = CgInput<'static>;
pub type ChandeInputS = ChandeInput<'static>;
pub type ChandelierExitInputS = ChandelierExitInput<'static>;
pub type ChopInputS = ChopInput<'static>;
pub type CkspInputS = CkspInput<'static>;
pub type CmoInputS = CmoInput<'static>;
pub type CoppockInputS = CoppockInput<'static>;
pub type CoraWaveInputS = CoraWaveInput<'static>;
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
pub type EhlersEcemaInputS = EhlersEcemaInput<'static>;
pub type EhlersITrendInputS = EhlersITrendInput<'static>;
pub type EhlersPmaInputS = EhlersPmaInput<'static>;
pub type EhlersKamaInputS = EhlersKamaInput<'static>;
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
pub type HighPassInputS = HighPassInput<'static>;
pub type HighPass2InputS = HighPass2Input<'static>;
pub type HmaInputS = HmaInput<'static>;
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
pub type LinearregAngleInputS = Linearreg_angleInput<'static>;
pub type LinearRegInterceptInputS = LinearRegInterceptInput<'static>;
pub type LinearRegSlopeInputS = LinearRegSlopeInput<'static>;
pub type LinRegInputS = LinRegInput<'static>;
pub type LpcInputS = LpcInput<'static>;
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
pub type ModGodModeInputS = ModGodModeInput<'static>;
pub type MomInputS = MomInput<'static>;
pub type MswInputS = MswInput<'static>;
pub type MwdxInputS = MwdxInput<'static>;
pub type NatrInputS = NatrInput<'static>;
pub type NweInputS = NweInput<'static>;
pub type NmaInputS = NmaInput<'static>;
pub type NviInputS = NviInput<'static>;
pub type ObvInputS = ObvInput<'static>;
pub type OttoInputS = OttoInput<'static>;
pub type OttInputS = OttInput<'static>;
pub type PfeInputS = PfeInput<'static>;
pub type PercentileNearestRankInputS = PercentileNearestRankInput<'static>;
pub type PivotInputS = PivotInput<'static>;
pub type PmaInputS = PmaInput<'static>;
pub type PpoInputS = PpoInput<'static>;
pub type PrbInputS = PrbInput<'static>;
pub type PviInputS = PviInput<'static>;
pub type PwmaInputS = PwmaInput<'static>;
pub type QqeInputS = QqeInput<'static>;
pub type QstickInputS = QstickInput<'static>;
pub type ReflexInputS = ReflexInput<'static>;
pub type RocInputS = RocInput<'static>;
pub type RocpInputS = RocpInput<'static>;
pub type RocrInputS = RocrInput<'static>;
pub type RsiInputS = RsiInput<'static>;
pub type RsmkInputS = RsmkInput<'static>;
pub type RsxInputS = RsxInput<'static>;
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
pub type StochInputS = StochInput<'static>;
pub type StochfInputS = StochfInput<'static>;
pub type SuperSmootherInputS = SuperSmootherInput<'static>;
pub type SupertrendInputS = SuperTrendInput<'static>;
pub type SuperSmoother3PoleInputS = SuperSmoother3PoleInput<'static>;
pub type SwmaInputS = SwmaInput<'static>;
pub type TemaInputS = TemaInput<'static>;
pub type TilsonInputS = TilsonInput<'static>;
pub type TradjemaInputS = TradjemaInput<'static>;
pub type TrendFlexInputS = TrendFlexInput<'static>;
pub type TrimaInputS = TrimaInput<'static>;
pub type UmaInputS = UmaInput<'static>;
pub type TrixInputS = TrixInput<'static>;
pub type TsfInputS = TsfInput<'static>;
pub type TsiInputS = TsiInput<'static>;
pub type TtmSqueezeInputS = TtmSqueezeInput<'static>;
pub type TtmTrendInputS = TtmTrendInput<'static>;
pub type UiInputS = UiInput<'static>;
pub type UltOscInputS = UltOscInput<'static>;
pub type VarInputS = VarInput<'static>;
pub type ViInputS = ViInput<'static>;
pub type VidyaInputS = VidyaInput<'static>;
pub type VlmaInputS = VlmaInput<'static>;
pub type VolumeAdjustedMaInputS = VolumeAdjustedMaInput<'static>;
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
pub type DeviationInputS = DeviationInput<'static>;
// ASO input alias
pub type AsoInputS = AsoInput<'static>;
// DVDIQQE single-series input type
pub type DvdiqqeInputS = DvdiqqeInput<'static>;

// Other indicators InputS types
pub type AvslInputS = AvslInput<'static>;
pub type DmaInputS = DmaInput<'static>;
pub type EhmaInputS = EhmaInput<'static>;
pub type RangeFilterInputS = RangeFilterInput<'static>;
pub type SamaInputS = SamaInput<'static>;
pub type WtoInputS = WtoInput<'static>;
pub type NamaInputS = NamaInput<'static>;
pub type VamaInputS = VamaInputMv<'static>;
pub type HalfTrendInputS = HalfTrendInput<'static>;
pub type NetMyrsiInputS = NetMyrsiInput<'static>;
pub type CciCycleInputS = CciCycleInput<'static>;
pub type FvgTrailingStopInputS = FvgTrailingStopInput<'static>;
pub type ReverseRsiInputS = ReverseRsiInput<'static>;
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
    window: Option<u64>, // ← renamed for clarity
) where
    F: Fn(&In) -> anyhow::Result<()> + Copy + 'static,
    In: InputLen + 'static,
{
    //------------------------------------------------------------------
    // 1️⃣  Tell Criterion how many **bytes** one iteration really moves
    //     • window == None   ➜ simple streaming pass:        len × 8
    //     • window == Some(w)➜ sliding window algorithm:
    //                         (w + 1) × len × 8  (reads + writes)
    //------------------------------------------------------------------
    let bytes_per_iter = match window {
        Some(w) => (len as u64) * (w + 1) * std::mem::size_of::<f64>() as u64,
        None => (len as u64) * std::mem::size_of::<f64>() as u64,
    };
    group.throughput(Throughput::Bytes(bytes_per_iter));

    //------------------------------------------------------------------
    // 2️⃣  Configure the group *before* registering the benchmark
    //------------------------------------------------------------------
    group.measurement_time(Duration::from_millis(900));
    group.warm_up_time(Duration::from_millis(150));
    group.sample_size(10);

    //------------------------------------------------------------------
    // 3️⃣  Register the benchmark
    //------------------------------------------------------------------
    let input = In::with_len(len);
    group.bench_with_input(
        BenchmarkId::new(label, pretty_len(len)),
        &input,
        move |b, input| b.iter(|| fun(black_box(input)).unwrap()),
    );
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

// Special implementation for MaczInputS
impl InputLen for MaczInputS {
    fn with_len(len: usize) -> Self {
        match len {
            10_000 => MaczInput::with_default_candles(&*CANDLES_10K),
            100_000 => MaczInput::with_default_candles(&*CANDLES_100K),
            1_000_000 => MaczInput::with_default_candles(&*CANDLES_1M),
            _ => panic!("unsupported len {len}"),
        }
    }
}

impl InputLen for RsmkInputS {
    fn with_len(len: usize) -> Self {
        match len {
            10_000 => RsmkInput::with_default_candles(&*CANDLES_10K, &*CANDLES_10K),
            100_000 => RsmkInput::with_default_candles(&*CANDLES_100K, &*CANDLES_100K),
            1_000_000 => RsmkInput::with_default_candles(&*CANDLES_1M, &*CANDLES_1M),
            _ => panic!("unsupported len {len}"),
        }
    }
}

// Special implementation for VwmaInputS which requires volume data
impl InputLen for VwmaInputS {
    fn with_len(len: usize) -> Self {
        match len {
            10_000 => VwmaInput::from_candles(&*CANDLES_10K, "close", VwmaParams::default()),
            100_000 => VwmaInput::from_candles(&*CANDLES_100K, "close", VwmaParams::default()),
            1_000_000 => VwmaInput::from_candles(&*CANDLES_1M, "close", VwmaParams::default()),
            _ => panic!("unsupported len {len}"),
        }
    }
}

impl_input_len!(
    AsoInputS,
    AcoscInputS,
    AdInputS,
    AdoscInputS,
    AdxInputS,
    AdxrInputS,
    AlligatorInputS,
    AlmaInputS,
    AlphaTrendInputS,
    AoInputS,
    ApoInputS,
    AroonInputS,
    AroonOscInputS,
    AtrInputS,
    BandPassInputS,
    BollingerBandsInputS,
    BollingerBandsWidthInputS,
    BopInputS,
    CciInputS,
    CfoInputS,
    CgInputS,
    ChandeInputS,
    ChandelierExitInputS,
    ChopInputS,
    CkspInputS,
    CmoInputS,
    CoppockInputS,
    CoraWaveInputS,
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
    EhlersEcemaInputS,
    EhlersITrendInputS,
    EhlersPmaInputS,
    EhlersKamaInputS,
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
    HighPassInputS,
    HighPass2InputS,
    HmaInputS,
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
    LinearregAngleInputS,
    LinearRegInterceptInputS,
    LinearRegSlopeInputS,
    LinRegInputS,
    LpcInputS,
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
    NviInputS,
    ObvInputS,
    OttoInputS,
    OttInputS,
    PercentileNearestRankInputS,
    PfeInputS,
    PivotInputS,
    PmaInputS,
    PpoInputS,
    PrbInputS,
    PviInputS,
    PwmaInputS,
    QstickInputS,
    ReflexInputS,
    RocInputS,
    RocpInputS,
    RocrInputS,
    RsiInputS,
    RsxInputS,
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
    StochInputS,
    StochfInputS,
    SuperSmootherInputS,
    SupertrendInputS,
    SuperSmoother3PoleInputS,
    SwmaInputS,
    TemaInputS,
    TilsonInputS,
    TradjemaInputS,
    TrendFlexInputS,
    TrimaInputS,
    TrixInputS,
    TsfInputS,
    TsiInputS,
    TtmTrendInputS,
    UiInputS,
    UltOscInputS,
    UmaInputS,
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
    VwmacdInputS,
    WadInputS,
    WavetrendInputS,
    WclpriceInputS,
    WildersInputS,
    WillrInputS,
    WmaInputS,
    ZlemaInputS,
    ZscoreInputS,
    // Other indicators
    AvslInputS,
    DmaInputS,
    EhmaInputS,
    RangeFilterInputS,
    SamaInputS,
    WtoInputS,
    NamaInputS,
    // Missing indicators
    ModGodModeInputS,
    NweInputS,
    QqeInputS,
    TtmSqueezeInputS,
    BuffAveragesInputS,
    VolumeAdjustedMaInputS,
    NetMyrsiInputS,
    CciCycleInputS,
    FvgTrailingStopInputS,
    HalfTrendInputS,
    ReverseRsiInputS,
    VamaInputS
);

bench_wrappers! {
    (acosc_bench, acosc_raw, AcoscInputS),
    (ad_bench, ad_raw, AdInputS),
    (adosc_bench, adosc_raw, AdoscInputS),
    (adx_bench, adx_raw, AdxInputS),
    (adxr_bench, adxr_raw, AdxrInputS),
    (alligator_bench, alligator_raw, AlligatorInputS),
    (alphatrend_bench, alphatrend_raw, AlphaTrendInputS),
    (ao_bench, ao_raw, AoInputS),
    (apo_bench, apo_raw, ApoInputS),
    (aroon_bench, aroon_raw, AroonInputS),
    (aroon_osc_bench, aroon_osc_raw, AroonOscInputS),
    (atr_bench, atr_raw, AtrInputS),
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
    (cora_wave_bench, cora_wave_raw, CoraWaveInputS),
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
    (ift_rsi_bench, ift_rsi_raw, IftRsiInputS),
    (kaufmanstop_bench, kaufmanstop_raw, KaufmanstopInputS),
    (kdj_bench, kdj_raw, KdjInputS),
    (keltner_bench, keltner_raw, KeltnerInputS),
    (kst_bench, kst_raw, KstInputS),
    (kurtosis_bench, kurtosis_raw, KurtosisInputS),
    (kvo_bench, kvo_raw, KvoInputS),
    (linearreg_angle_bench, linearreg_angle_raw, LinearregAngleInputS),
    (linearreg_intercept_bench, linearreg_intercept_raw, LinearRegInterceptInputS),
    (linearreg_slope_bench, linearreg_slope_raw, LinearRegSlopeInputS),
    (lpc_bench, lpc_raw, LpcInputS),
    (lrsi_bench, lrsi_raw, LrsiInputS),
    (mab_bench, mab_raw, MabInputS),
    (macd_bench, macd_raw, MacdInputS),
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
    (natr_bench, natr_raw, NatrInputS),
    (nvi_bench, nvi_raw, NviInputS),
    (obv_bench, obv_raw, ObvInputS),
    (ott_bench, ott_raw, OttInputS),
    (otto_bench, otto_raw, OttoInputS),
    (pfe_bench, pfe_raw, PfeInputS),
    (pivot_bench, pivot_raw, PivotInputS),
    (pma_bench, pma_raw, PmaInputS),
    (ppo_bench, ppo_raw, PpoInputS),
    (prb_bench, prb_raw, PrbInputS),
    (pvi_bench, pvi_raw, PviInputS),
    (qstick_bench, qstick_raw, QstickInputS),
    (roc_bench, roc_raw, RocInputS),
    (rocp_bench, rocp_raw, RocpInputS),
    (rocr_bench, rocr_raw, RocrInputS),
    (rsi_bench, rsi_raw, RsiInputS),
    (rsmk_bench, rsmk_raw, RsmkInputS),
    (rsx_bench, rsx_raw, RsxInputS),
    (rvi_bench, rvi_raw, RviInputS),
    (safezonestop_bench, safezonestop_raw, SafeZoneStopInputS),
    (sar_bench, sar_raw, SarInputS),
    (squeeze_momentum_bench, squeeze_momentum_raw, SqueezeMomentumInputS),
    (srsi_bench, srsi_raw, SrsiInputS),
    (stc_bench, stc_raw, StcInputS),
    (stddev_bench, stddev_raw, StdDevInputS),
    (stoch_bench, stoch_raw, StochInputS),
    (stochf_bench, stochf_raw, StochfInputS),
    (supertrend_bench, supertrend_raw, SupertrendInputS),
    (tsf_bench, tsf_raw, TsfInputS),
    (tsi_bench, tsi_raw, TsiInputS),
    (ttm_trend_bench, ttm_trend_raw, TtmTrendInputS),
    (ttm_squeeze_bench, ttm_squeeze_raw, TtmSqueezeInputS),
    (ui_bench, ui_raw, UiInputS),
    (ultosc_bench, ultosc_raw, UltOscInputS),
    (var_bench, var_raw, VarInputS),
    (vi_bench, vi_raw, ViInputS),
    (vosc_bench, vosc_raw, VoscInputS),
    (voss_bench, voss_raw, VossInputS),
    (vpci_bench, vpci_raw, VpciInputS),
    (vpt_bench, vpt_raw, VptInputS),
    (vwap_bench, vwap_raw, VwapInputS),
    (vwmacd_bench, vwmacd_raw, VwmacdInputS),
    (wad_bench, wad_raw, WadInputS),
    (wavetrend_bench, wavetrend_raw, WavetrendInputS),
    (wclprice_bench, wclprice_raw, WclpriceInputS),
    (willr_bench, willr_raw, WillrInputS),
    (zscore_bench, zscore_raw, ZscoreInputS),
    // Missing indicators - create wrapper functions with different names
    (mod_god_mode_bench, mod_god_mode_raw, ModGodModeInputS),
    (nadaraya_watson_envelope_bench, nadaraya_watson_envelope_raw, NweInputS),
    (qqe_bench, qqe_raw, QqeInputS),
    (buff_averages_bench, buff_averages, BuffAveragesInputS),
    (volume_adjusted_ma_bench, VolumeAdjustedMa, VolumeAdjustedMaInputS),
    (net_myrsi_bench, net_myrsi, NetMyrsiInputS),
    (cci_cycle_bench, cci_cycle, CciCycleInputS),
    (fvg_trailing_stop_bench, fvg_trailing_stop, FvgTrailingStopInputS),
    (halftrend_bench, halftrend, HalfTrendInputS),
    (reverse_rsi_bench, reverse_rsi, ReverseRsiInputS),
    (vama_bench, vama, VamaInputS),
}

bench_scalars!(
    acosc_bench => AcoscInputS,
    ad_bench    => AdInputS,
    adosc_bench => AdoscInputS,
    adx_bench   => AdxInputS,
    adxr_bench  => AdxrInputS,
    alligator_bench => AlligatorInputS,
    alphatrend_bench => AlphaTrendInputS,

    ao_bench   => AoInputS,
    apo_bench  => ApoInputS,

    aroon_bench        => AroonInputS,
    aroon_osc_bench    => AroonOscInputS,
    atr_bench          => AtrInputS,
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
    cora_wave_bench   => CoraWaveInputS,
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
    ift_rsi_bench     => IftRsiInputS,
    kaufmanstop_bench => KaufmanstopInputS,
    kdj_bench         => KdjInputS,
    keltner_bench     => KeltnerInputS,
    kst_bench         => KstInputS,
    kurtosis_bench    => KurtosisInputS,
    kvo_bench         => KvoInputS,

    linearreg_angle_bench     => LinearregAngleInputS,
    linearreg_intercept_bench => LinearRegInterceptInputS,
    linearreg_slope_bench     => LinearRegSlopeInputS,
    lpc_bench                 => LpcInputS,
    lrsi_bench                => LrsiInputS,

    mab_bench  => MabInputS,
    macd_bench => MacdInputS,
    marketfi_bench  => MarketefiInputS,
    mass_bench      => MassInputS,
    mean_ad_bench   => MeanAdInputS,
    medium_ad_bench => MediumAdInputS,
    medprice_bench  => MedpriceInputS,
    mfi_bench       => MfiInputS,
    midpoint_bench  => MidpointInputS,
    midprice_bench  => MidpriceInputS,
    minmax_bench    => MinmaxInputS,
    mod_god_mode_bench => ModGodModeInputS,
    mom_bench       => MomInputS,
    msw_bench       => MswInputS,
    nadaraya_watson_envelope_bench => NweInputS,

    natr_bench   => NatrInputS,
    nvi_bench    => NviInputS,
    obv_bench    => ObvInputS,
    ott_bench    => OttInputS,
    otto_bench   => OttoInputS,
    pfe_bench    => PfeInputS,
    pivot_bench  => PivotInputS,
    pma_bench    => PmaInputS,
    ppo_bench    => PpoInputS,
    prb_bench    => PrbInputS,
    pvi_bench    => PviInputS,
    qqe_bench    => QqeInputS,
    qstick_bench => QstickInputS,
    roc_bench    => RocInputS,
    rocp_bench   => RocpInputS,
    rocr_bench   => RocrInputS,
    rsi_bench    => RsiInputS,
    rsmk_bench   => RsmkInputS,
    rsx_bench    => RsxInputS,
    rvi_bench    => RviInputS,
    safezonestop_bench => SafeZoneStopInputS,
    sar_bench    => SarInputS,
    squeeze_momentum_bench => SqueezeMomentumInputS,
    srsi_bench   => SrsiInputS,
    stc_bench    => StcInputS,
    stddev_bench => StdDevInputS,
    stoch_bench  => StochInputS,
    stochf_bench => StochfInputS,
    supertrend_bench => SupertrendInputS,
    tsf_bench    => TsfInputS,
    tsi_bench    => TsiInputS,
    ttm_squeeze_bench => TtmSqueezeInputS,
    ttm_trend_bench => TtmTrendInputS,
    ui_bench     => UiInputS,
    ultosc_bench => UltOscInputS,
    var_bench    => VarInputS,
    vi_bench     => ViInputS,
    vosc_bench   => VoscInputS,
    voss_bench   => VossInputS,
    vpci_bench   => VpciInputS,
    vpt_bench    => VptInputS,
    vwap_bench   => VwapInputS,
    vwmacd_bench => VwmacdInputS,
    wad_bench    => WadInputS,
    wavetrend_bench => WavetrendInputS,
    wclprice_bench => WclpriceInputS,
    willr_bench     => WillrInputS,
    zscore_bench    => ZscoreInputS,
    // Missing indicators
    buff_averages_bench => BuffAveragesInputS,
    volume_adjusted_ma_bench => VolumeAdjustedMaInputS,
    net_myrsi_bench => NetMyrsiInputS,
    cci_cycle_bench => CciCycleInputS,
    fvg_trailing_stop_bench => FvgTrailingStopInputS,
    halftrend_bench => HalfTrendInputS,
    reverse_rsi_bench => ReverseRsiInputS,
    vama_bench => VamaInputS
);

make_kernel_wrappers!(alma, alma_with_kernel, AlmaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(bandpass, bandpass_with_kernel, BandPassInputS; Scalar,Avx2,Avx512);
// ACOSC single-series kernels (Scalar/Avx2/Avx512)
make_kernel_wrappers!(
    acosc,
    my_project::indicators::acosc::acosc_with_kernel,
    AcoscInputS;
    Scalar,Avx2,Avx512
);
make_kernel_wrappers!(cfo, cfo_with_kernel, CfoInputS; Scalar,Avx2,Avx512);
// AD single-series wrappers (Scalar/AVX2/AVX512)
make_kernel_wrappers!(ad, ad_with_kernel, AdInputS; Scalar,Avx2,Avx512);
// ADOSC single-series variants (Scalar, AVX2, AVX512)
make_kernel_wrappers!(
    adosc,
    my_project::indicators::adosc::adosc_with_kernel,
    AdoscInputS;
    Scalar,Avx2,Avx512
);
// AO single-series kernel wrappers (Scalar/AVX2/AVX512)
make_kernel_wrappers!(
    ao,
    my_project::indicators::ao::ao_with_kernel,
    AoInputS;
    Scalar,Avx2,Avx512
);
// ATR single-series kernel variants (Scalar, AVX2, AVX512)
make_kernel_wrappers!(atr, my_project::indicators::atr::atr_with_kernel, AtrInputS; Scalar,Avx2,Avx512);
// DVDIQQE single-series kernel variants (Scalar, AVX2, AVX512)
make_kernel_wrappers!(dvdiqqe, dvdiqqe_with_kernel, DvdiqqeInputS; Scalar,Avx2,Avx512);
// MACD single-series kernel variants (Scalar, AVX2, AVX512)
make_kernel_wrappers!(macd, my_project::indicators::macd::macd_with_kernel, MacdInputS; Scalar,Avx2,Avx512);
// MSW single-series kernel variants (Scalar, AVX2, AVX512)
make_kernel_wrappers!(msw, my_project::indicators::msw::msw_with_kernel, MswInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(adx, adx_with_kernel, AdxInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(buff_averages, buff_averages_with_kernel, BuffAveragesInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(correl_hl, correl_hl_with_kernel, CorrelHlInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(zscore, zscore_with_kernel, ZscoreInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(var, var_with_kernel, VarInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(cmo, cmo_with_kernel, CmoInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(srsi, srsi_with_kernel, SrsiInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(er, er_with_kernel, ErInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(macz, macz_with_kernel, MaczInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(mom, mom_with_kernel, MomInputS; Scalar,Avx2,Avx512);
// ASO single-series wrappers (Scalar/AVX2/AVX512)
make_kernel_wrappers!(aso, aso_with_kernel, AsoInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(cwma, cwma_with_kernel, CwmaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(natr, natr_with_kernel, NatrInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(marketfi, marketefi_with_kernel, MarketefiInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(efi, efi_with_kernel, EfiInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(dema, dema_with_kernel, DemaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(cci, cci_with_kernel, CciInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(edcf, edcf_with_kernel, EdcfInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(ehlers_ecema, ehlers_ecema_with_kernel, EhlersEcemaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(ehlers_itrend, ehlers_itrend_with_kernel, EhlersITrendInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(ehlers_pma, ehlers_pma_with_kernel, EhlersPmaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(ehlers_kama, ehlers_kama_with_kernel, EhlersKamaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(ema, ema_with_kernel, EmaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(epma, epma_with_kernel, EpmaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(pma, pma_with_kernel, PmaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(frama, frama_with_kernel, FramaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(fwma, fwma_with_kernel, FwmaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(gaussian, gaussian_with_kernel, GaussianInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(highpass_2_pole, highpass_2_pole_with_kernel, HighPass2InputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(highpass, highpass_with_kernel, HighPassInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(hma, hma_with_kernel, HmaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(hwma, hwma_with_kernel, HwmaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(jma, jma_with_kernel, JmaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(jsa, jsa_with_kernel, JsaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(kama, kama_with_kernel, KamaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(linreg, linreg_with_kernel, LinRegInputS; Scalar,Avx2,Avx512);
// linearreg_angle: expose scalar/avx2/avx512 wrappers for variant benches
make_kernel_wrappers!(linearreg_angle, linearreg_angle_with_kernel, LinearregAngleInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(
    linearreg_intercept,
    my_project::indicators::linearreg_intercept::linearreg_intercept_with_kernel,
    LinearRegInterceptInputS;
    Scalar,Avx2,Avx512
);
make_kernel_wrappers!(maaq, maaq_with_kernel, MaaqInputS; Scalar,Avx2,Avx512);
// MAB: kernel-specific wrappers for SIMD benchmarking
make_kernel_wrappers!(mab, mab_with_kernel, MabInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(mama, mama_with_kernel, MamaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(mwdx, mwdx_with_kernel, MwdxInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(nma, nma_with_kernel, NmaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(pwma, pwma_with_kernel, PwmaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(reflex, reflex_with_kernel, ReflexInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(sinwma, sinwma_with_kernel, SinWmaInputS; Scalar,Avx2,Avx512);
// Enable kernel variant benches for mod_god_mode
make_kernel_wrappers!(mod_god_mode, mod_god_mode_with_kernel, ModGodModeInputS; Scalar,Avx2,Avx512);
 make_kernel_wrappers!(fosc, fosc_with_kernel, FoscInputS; Scalar,Avx2,Avx512);
 make_kernel_wrappers!(dpo, dpo_with_kernel, DpoInputS; Scalar,Avx2,Avx512);
 make_batch_wrappers!(dpo_batch, DpoBatchBuilder, DpoInputS; ScalarBatch,Avx2Batch,Avx512Batch);
make_kernel_wrappers!(kaufmanstop, kaufmanstop_with_kernel, KaufmanstopInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(cksp, cksp_with_kernel, CkspInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(sma, sma_with_kernel, SmaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(smma, smma_with_kernel, SmmaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(sqwma, sqwma_with_kernel, SqwmaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(srwma, srwma_with_kernel, SrwmaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(supersmoother, supersmoother_with_kernel, SuperSmootherInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(supersmoother_3_pole, supersmoother_3_pole_with_kernel, SuperSmoother3PoleInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(swma, swma_with_kernel, SwmaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(tema, tema_with_kernel, TemaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(tilson, tilson_with_kernel, TilsonInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(tradjema, tradjema_with_kernel, TradjemaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(trendflex, trendflex_with_kernel, TrendFlexInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(trima, trima_with_kernel, TrimaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(uma, uma_with_kernel, UmaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(chandelier_exit, chandelier_exit_with_kernel, ChandelierExitInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(percentile_nearest_rank, percentile_nearest_rank_with_kernel, PercentileNearestRankInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(vidya, vidya_with_kernel, VidyaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(volume_adjusted_ma, VolumeAdjustedMa_with_kernel, VolumeAdjustedMaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(vlma, vlma_with_kernel, VlmaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(vpwma, vpwma_with_kernel, VpwmaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(vwma, vwma_with_kernel, VwmaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(wclprice, wclprice_with_kernel, WclpriceInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(wilders, wilders_with_kernel, WildersInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(wma, wma_with_kernel, WmaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(zlema, zlema_with_kernel, ZlemaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(sar, sar_with_kernel, SarInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(ppo, ppo_with_kernel, PpoInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(kurtosis, kurtosis_with_kernel, KurtosisInputS; Scalar,Avx2,Avx512);
// ADXR single-series kernel wrappers (add explicit kernel variants)
make_kernel_wrappers!(adxr, adxr_with_kernel, AdxrInputS; Scalar,Avx2,Avx512);

// Stochastic Oscillator (single-series) kernel wrappers
make_kernel_wrappers!(stoch, stoch_with_kernel, StochInputS; Scalar,Avx2,Avx512);

// Stochastic Oscillator (batch) custom wrappers (OHLC inputs)
#[inline(always)]
fn stoch_batch_scalarbatch(input: &StochInputS) -> anyhow::Result<()> {
    let (high, low, close) = match &input.data {
        StochData::Candles { candles } => (&candles.high[..], &candles.low[..], &candles.close[..]),
        StochData::Slices { high, low, close } => (*high, *low, *close),
    };
    StochBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .apply_slices(high, low, close)?;
    Ok(())
}

// SAR (batch) custom wrappers (high/low inputs)
#[inline(always)]
fn sar_batch_scalarbatch(input: &SarInputS) -> anyhow::Result<()> {
    let (high, low) = match &input.data {
        SarData::Candles { candles } => (&candles.high[..], &candles.low[..]),
        SarData::Slices { high, low } => (*high, *low),
    };
    SarBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .apply_slices(high, low)
        .map(|_| ())
        .map_err(|e| anyhow!(e.to_string()))
}

#[inline(always)]
fn sar_batch_avx2batch(input: &SarInputS) -> anyhow::Result<()> {
    let (high, low) = match &input.data {
        SarData::Candles { candles } => (&candles.high[..], &candles.low[..]),
        SarData::Slices { high, low } => (*high, *low),
    };
    SarBatchBuilder::new()
        .kernel(Kernel::Avx2Batch)
        .apply_slices(high, low)
        .map(|_| ())
        .map_err(|e| anyhow!(e.to_string()))
}

#[inline(always)]
fn sar_batch_avx512batch(input: &SarInputS) -> anyhow::Result<()> {
    let (high, low) = match &input.data {
        SarData::Candles { candles } => (&candles.high[..], &candles.low[..]),
        SarData::Slices { high, low } => (*high, *low),
    };
    SarBatchBuilder::new()
        .kernel(Kernel::Avx512Batch)
        .apply_slices(high, low)
        .map(|_| ())
        .map_err(|e| anyhow!(e.to_string()))
}

#[inline(always)]
fn stoch_batch_avx2batch(input: &StochInputS) -> anyhow::Result<()> {
    let (high, low, close) = match &input.data {
        StochData::Candles { candles } => (&candles.high[..], &candles.low[..], &candles.close[..]),
        StochData::Slices { high, low, close } => (*high, *low, *close),
    };
    StochBatchBuilder::new()
        .kernel(Kernel::Avx2Batch)
        .apply_slices(high, low, close)?;
    Ok(())
}

#[inline(always)]
fn stoch_batch_avx512batch(input: &StochInputS) -> anyhow::Result<()> {
    let (high, low, close) = match &input.data {
        StochData::Candles { candles } => (&candles.high[..], &candles.low[..], &candles.close[..]),
        StochData::Slices { high, low, close } => (*high, *low, *close),
    };
    StochBatchBuilder::new()
        .kernel(Kernel::Avx512Batch)
        .apply_slices(high, low, close)?;
    Ok(())
}

// Fisher (batch) custom wrappers (high/low inputs)
#[inline(always)]
fn fisher_batch_scalarbatch(input: &FisherInputS) -> anyhow::Result<()> {
    use my_project::indicators::fisher::{FisherBatchBuilder, FisherData};
    let (high, low) = match &input.data {
        FisherData::Candles { candles } => (&candles.high[..], &candles.low[..]),
        FisherData::Slices { high, low } => (*high, *low),
    };
    FisherBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .apply_slices(high, low)
        .map(|_| ())
        .map_err(|e| anyhow!(e.to_string()))
}

#[inline(always)]
fn fisher_batch_avx2batch(input: &FisherInputS) -> anyhow::Result<()> {
    use my_project::indicators::fisher::{FisherBatchBuilder, FisherData};
    let (high, low) = match &input.data {
        FisherData::Candles { candles } => (&candles.high[..], &candles.low[..]),
        FisherData::Slices { high, low } => (*high, *low),
    };
    FisherBatchBuilder::new()
        .kernel(Kernel::Avx2Batch)
        .apply_slices(high, low)
        .map(|_| ())
        .map_err(|e| anyhow!(e.to_string()))
}

#[inline(always)]
fn fisher_batch_avx512batch(input: &FisherInputS) -> anyhow::Result<()> {
    use my_project::indicators::fisher::{FisherBatchBuilder, FisherData};
    let (high, low) = match &input.data {
        FisherData::Candles { candles } => (&candles.high[..], &candles.low[..]),
        FisherData::Slices { high, low } => (*high, *low),
    };
    FisherBatchBuilder::new()
        .kernel(Kernel::Avx512Batch)
        .apply_slices(high, low)
        .map(|_| ())
        .map_err(|e| anyhow!(e.to_string()))
}
make_kernel_wrappers!(deviation, deviation_with_kernel, DeviationInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(correlation_cycle, correlation_cycle_with_kernel, CorrelationCycleInputS; Scalar,Avx2,Avx512);
// StochF single-series kernels
make_kernel_wrappers!(stochf, stochf_with_kernel, StochfInputS; Scalar,Avx2,Avx512);

// ACOSC kernel variant benches (Scalar/Avx2/Avx512)
bench_variants!(
    acosc => AcoscInputS; None;
    acosc_scalar,
    acosc_avx2,
    acosc_avx512
);

// Bollinger Bands Width single-series kernel variants
make_kernel_wrappers!(
    bollinger_bands_width,
    bollinger_bands_width_with_kernel,
    BollingerBandsWidthInputS;
    Scalar,Avx2,Avx512
);
make_kernel_wrappers!(pvi, pvi_with_kernel, PviInputS; Scalar,Avx2,Avx512);
// DTI: single-series kernel wrappers for SIMD benchmarking
make_kernel_wrappers!(dti, dti_with_kernel, DtiInputS; Scalar,Avx2,Avx512);

// EMV: single-series kernel wrappers (SIMD stubs delegate to scalar)
make_kernel_wrappers!(emv, emv_with_kernel, EmvInputS; Scalar,Avx2,Avx512);

// NVI: kernel-specific wrappers for SIMD benchmarking
make_kernel_wrappers!(nvi, nvi_with_kernel, NviInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(trix, trix_with_kernel, TrixInputS; Scalar,Avx2,Avx512);
// Medium AD: enable Scalar vs AVX2 vs AVX512 comparisons
make_kernel_wrappers!(
    medium_ad,
    my_project::indicators::medium_ad::medium_ad_with_kernel,
    MediumAdInputS;
    Scalar,Avx2,Avx512
);
// Pivot single-series kernel wrappers
make_kernel_wrappers!(pivot, pivot_with_kernel, PivotInputS; Scalar,Avx2,Avx512);

// ROCP (Rate of Change Percentage) single-series wrappers
make_kernel_wrappers!(rocp, my_project::indicators::rocp::rocp_with_kernel, RocpInputS; Scalar,Avx2,Avx512);

// STC (single-series) kernel wrappers for Scalar/AVX2/AVX512
make_kernel_wrappers!(stc, stc_with_kernel, StcInputS; Scalar,Avx2,Avx512);

// Damiani Volatmeter single-series variants (Scalar/AVX2/AVX512)
make_kernel_wrappers!(
    damiani_volatmeter,
    my_project::indicators::damiani_volatmeter::damiani_volatmeter_with_kernel,
    DamianiVolatmeterInputS;
    Scalar,Avx2,Avx512
);
make_kernel_wrappers!(tsf, tsf_with_kernel, TsfInputS; Scalar,Avx2,Avx512);
// Add ROCR kernel-specific wrappers to compare scalar vs SIMD
make_kernel_wrappers!(rocr, rocr_with_kernel, RocrInputS; Scalar,Avx2,Avx512);
// OTT kernel-specific wrappers to compare scalar vs SIMD
make_kernel_wrappers!(ott, ott_with_kernel, OttInputS; Scalar,Avx2,Avx512);
// LinearRegSlope kernel-specific wrappers to compare scalar vs SIMD
make_kernel_wrappers!(
    linearreg_slope,
    linearreg_slope_with_kernel,
    LinearRegSlopeInputS;
    Scalar,Avx2,Avx512
);

// APO kernel-specific wrappers to compare scalar vs SIMD
make_kernel_wrappers!(apo, apo_with_kernel, ApoInputS; Scalar,Avx2,Avx512);

// Other indicators kernel wrappers
make_kernel_wrappers!(
    bollinger_bands,
    bollinger_bands_with_kernel,
    BollingerBandsInputS;
    Scalar,Avx2,Avx512
);
make_kernel_wrappers!(avsl, avsl_with_kernel, AvslInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(dma, dma_with_kernel, DmaInputS; Scalar,Avx2,Avx512);
// DM kernel-specific wrappers to compare scalar vs SIMD
make_kernel_wrappers!(dm, dm_with_kernel, DmInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(ehma, ehma_with_kernel, EhmaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(range_filter, range_filter_with_kernel, RangeFilterInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(sama, sama_with_kernel, SamaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(wto, wto_with_kernel, WtoInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(nama, nama_with_kernel, NamaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(net_myrsi, net_myrsi_with_kernel, NetMyrsiInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(cci_cycle, cci_cycle_with_kernel, CciCycleInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(fvg_trailing_stop, fvg_trailing_stop_with_kernel, FvgTrailingStopInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(halftrend, halftrend_with_kernel, HalfTrendInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(reverse_rsi, reverse_rsi_with_kernel, ReverseRsiInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(vama, vama_with_kernel, VamaInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(wavetrend, wavetrend_with_kernel, WavetrendInputS; Scalar,Avx2,Avx512);

// Donchian: add explicit kernel variants alongside the scalar-only bench
make_kernel_wrappers!(donchian, donchian_with_kernel, DonchianInputS; Scalar,Avx2,Avx512);

// RSX single-series variants
make_kernel_wrappers!(rsx, rsx_with_kernel, RsxInputS; Scalar,Avx2,Avx512);


// RSX batch variants
make_batch_wrappers!(
    rsx_batch, RsxBatchBuilder, RsxInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

// CG: kernel-specific wrappers for SIMD benchmarking
make_kernel_wrappers!(cg, cg_with_kernel, CgInputS; Scalar,Avx2,Avx512);

// CG: compare scalar vs AVX2 vs AVX512 at multiple sizes
bench_variants!(
    cg => CgInputS; Some(227);
    cg_scalar,
    cg_avx2,
    cg_avx512,
);

// AD single-series variants
bench_variants!(
    ad => AdInputS; None;
    ad_scalar,
    ad_avx2,
    ad_avx512,
);

// Bollinger Bands single-series variants
bench_variants!(
    bollinger_bands => BollingerBandsInputS; Some(20);
    bollinger_bands_scalar,
    bollinger_bands_avx2,
    bollinger_bands_avx512,
);

// ASO single-series variants (window ~ default period 10)
bench_variants!(
    aso => AsoInputS; Some(10);
    aso_scalar,
    aso_avx2,
    aso_avx512,
);

// AlphaTrend single-series variants
make_kernel_wrappers!(alphatrend, alphatrend_with_kernel, AlphaTrendInputS; Scalar,Avx2,Avx512);
bench_variants!(
    alphatrend => AlphaTrendInputS; None;
    alphatrend_scalar,
    alphatrend_avx2,
    alphatrend_avx512,
);


make_kernel_wrappers!(supertrend, supertrend_with_kernel, SupertrendInputS; Scalar,Avx2,Avx512);
make_kernel_wrappers!(cvi, cvi_with_kernel, CviInputS; Scalar,Avx2,Avx512);

// Bollinger Bands batch wrappers
make_batch_wrappers!(
    bollinger_bands_batch, BollingerBandsBatchBuilder, BollingerBandsInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

// Bollinger Bands batch variants
bench_variants!(
    bollinger_bands_batch => BollingerBandsInputS; None;
    bollinger_bands_batch_scalarbatch,
    bollinger_bands_batch_avx2batch,
    bollinger_bands_batch_avx512batch,
);

// ASO batch variants
bench_variants!(
    aso_batch => AsoInputS; None;
    aso_batch_scalarbatch,
    aso_batch_avx2batch,
    aso_batch_avx512batch,
);
// Ulcer Index (UI) single-series variants
make_kernel_wrappers!(ui, ui_with_kernel, UiInputS; Scalar,Avx2,Avx512);
// Gator Oscillator single-series variants (Scalar, AVX2, AVX512)
make_kernel_wrappers!(
    gatorosc,
    my_project::indicators::gatorosc::gatorosc_with_kernel,
    GatorOscInputS;
    Scalar,Avx2,Avx512
);
// OBV single-series variants
make_kernel_wrappers!(
    obv,
    my_project::indicators::obv::obv_with_kernel,
    ObvInputS;
    Scalar,Avx2,Avx512
);

// Coppock single-series variants (Scalar, AVX2, AVX512)
make_kernel_wrappers!(
    coppock,
    my_project::indicators::coppock::coppock_with_kernel,
    CoppockInputS;
    Scalar,Avx2,Avx512
);

// BOP single-series variants (Scalar, AVX2, AVX512)
make_kernel_wrappers!(
    bop,
    my_project::indicators::bop::bop_with_kernel,
    BopInputS;
    Scalar,Avx2,Avx512
);
make_kernel_wrappers!(ultosc, ultosc_with_kernel, UltOscInputS; Scalar,Avx2,Avx512);
 make_kernel_wrappers!(qstick, qstick_with_kernel, QstickInputS; Scalar,Avx2,Avx512);
 make_kernel_wrappers!(nadaraya_watson_envelope, nadaraya_watson_envelope_with_kernel, NweInputS; Scalar,Avx2,Avx512);
 make_kernel_wrappers!(eri, eri_with_kernel, EriInputS; Scalar,Avx2,Avx512);

// VOSC batch wrappers
make_batch_wrappers!(
    vosc_batch, VoscBatchBuilder, VoscInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

// MFI batch wrappers
#[inline(always)]
fn mfi_batch_scalarbatch(input: &MfiInputS) -> anyhow::Result<()> {
    let (tp, vol) = match &input.data {
        MfiData::Candles { candles, source } => (source_type(candles, source), &candles.volume[..]),
        MfiData::Slices { typical_price, volume } => (*typical_price, *volume),
    };
    // Wider sweep to exercise shared precompute path
    let sweep = MfiBatchRange { period: (5, 200, 5) };
    MfiBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .period_range(sweep.period.0, sweep.period.1, sweep.period.2)
        .apply_slices(tp, vol)?;
    Ok(())
}

// DX batch wrappers
#[inline(always)]
fn dx_batch_scalarbatch(input: &DxInputS) -> anyhow::Result<()> {
    let (high, low, close) = match &input.data {
        my_project::indicators::dx::DxData::Candles { candles } => (
            source_type(candles, "high"),
            source_type(candles, "low"),
            source_type(candles, "close"),
        ),
        my_project::indicators::dx::DxData::HlcSlices { high, low, close } => (*high, *low, *close),
    };
    DxBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .period_range(10, 30, 5)
        .apply_hlc(high, low, close)
        .map(|_| ())
        .map_err(|e| anyhow!(e.to_string()))
}

// ADX batch wrappers (H/L/C inputs)
#[inline(always)]
fn adx_batch_scalarbatch(input: &AdxInputS) -> anyhow::Result<()> {
    let (high, low, close) = match &input.data {
        my_project::indicators::adx::AdxData::Candles { candles } => (
            source_type(candles, "high"),
            source_type(candles, "low"),
            source_type(candles, "close"),
        ),
        my_project::indicators::adx::AdxData::Slices { high, low, close } => (*high, *low, *close),
    };
    AdxBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .period_range(10, 30, 5)
        .apply_slices(high, low, close)
        .map(|_| ())
        .map_err(|e| anyhow!(e.to_string()))
}

#[inline(always)]
fn adx_batch_avx2batch(input: &AdxInputS) -> anyhow::Result<()> {
    let (high, low, close) = match &input.data {
        my_project::indicators::adx::AdxData::Candles { candles } => (
            source_type(candles, "high"),
            source_type(candles, "low"),
            source_type(candles, "close"),
        ),
        my_project::indicators::adx::AdxData::Slices { high, low, close } => (*high, *low, *close),
    };
    AdxBatchBuilder::new()
        .kernel(Kernel::Avx2Batch)
        .period_range(10, 30, 5)
        .apply_slices(high, low, close)
        .map(|_| ())
        .map_err(|e| anyhow!(e.to_string()))
}

#[inline(always)]
fn adx_batch_avx512batch(input: &AdxInputS) -> anyhow::Result<()> {
    let (high, low, close) = match &input.data {
        my_project::indicators::adx::AdxData::Candles { candles } => (
            source_type(candles, "high"),
            source_type(candles, "low"),
            source_type(candles, "close"),
        ),
        my_project::indicators::adx::AdxData::Slices { high, low, close } => (*high, *low, *close),
    };
    AdxBatchBuilder::new()
        .kernel(Kernel::Avx512Batch)
        .period_range(10, 30, 5)
        .apply_slices(high, low, close)
        .map(|_| ())
        .map_err(|e| anyhow!(e.to_string()))
}

#[inline(always)]
fn dx_batch_avx2batch(input: &DxInputS) -> anyhow::Result<()> {
    let (high, low, close) = match &input.data {
        my_project::indicators::dx::DxData::Candles { candles } => (
            source_type(candles, "high"),
            source_type(candles, "low"),
            source_type(candles, "close"),
        ),
        my_project::indicators::dx::DxData::HlcSlices { high, low, close } => (*high, *low, *close),
    };
    DxBatchBuilder::new()
        .kernel(Kernel::Avx2Batch)
        .period_range(10, 30, 5)
        .apply_hlc(high, low, close)
        .map(|_| ())
        .map_err(|e| anyhow!(e.to_string()))
}

#[inline(always)]
fn dx_batch_avx512batch(input: &DxInputS) -> anyhow::Result<()> {
    let (high, low, close) = match &input.data {
        my_project::indicators::dx::DxData::Candles { candles } => (
            source_type(candles, "high"),
            source_type(candles, "low"),
            source_type(candles, "close"),
        ),
        my_project::indicators::dx::DxData::HlcSlices { high, low, close } => (*high, *low, *close),
    };
    DxBatchBuilder::new()
        .kernel(Kernel::Avx512Batch)
        .period_range(10, 30, 5)
        .apply_hlc(high, low, close)
        .map(|_| ())
        .map_err(|e| anyhow!(e.to_string()))
}

#[inline(always)]
fn mfi_batch_avx2batch(input: &MfiInputS) -> anyhow::Result<()> {
    let (tp, vol) = match &input.data {
        MfiData::Candles { candles, source } => (source_type(candles, source), &candles.volume[..]),
        MfiData::Slices { typical_price, volume } => (*typical_price, *volume),
    };
    let sweep = MfiBatchRange { period: (5, 200, 5) };
    MfiBatchBuilder::new()
        .kernel(Kernel::Avx2Batch)
        .period_range(sweep.period.0, sweep.period.1, sweep.period.2)
        .apply_slices(tp, vol)?;
    Ok(())
}

#[inline(always)]
fn mfi_batch_avx512batch(input: &MfiInputS) -> anyhow::Result<()> {
    let (tp, vol) = match &input.data {
        MfiData::Candles { candles, source } => (source_type(candles, source), &candles.volume[..]),
        MfiData::Slices { typical_price, volume } => (*typical_price, *volume),
    };
    let sweep = MfiBatchRange { period: (5, 200, 5) };
    MfiBatchBuilder::new()
        .kernel(Kernel::Avx512Batch)
        .period_range(sweep.period.0, sweep.period.1, sweep.period.2)
        .apply_slices(tp, vol)?;
    Ok(())
}

make_batch_wrappers!(
    apo_batch, ApoBatchBuilder, ApoInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

bench_variants!(
    apo_batch => ApoInputS; None;
    apo_batch_scalarbatch,
    apo_batch_avx2batch,
    apo_batch_avx512batch,
);

make_batch_wrappers!(
    alma_batch, AlmaBatchBuilder, AlmaInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

// ADOSC batch wrappers (ScalarBatch/Avx2Batch/Avx512Batch)
// ADOSC batch wrappers require OHLCV slices
#[inline(always)]
fn adosc_batch_scalarbatch(input: &AdoscInputS) -> anyhow::Result<()> {
    use my_project::indicators::adosc::AdoscBatchBuilder;
    let (h, l, c, v) = match &input.data {
        my_project::indicators::adosc::AdoscData::Candles { candles } => (
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            candles.volume.as_slice(),
        ),
        my_project::indicators::adosc::AdoscData::Slices { high, low, close, volume } =>
            (*high, *low, *close, *volume),
    };
    AdoscBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .apply_slices(h, l, c, v)
        .map(|_| ())
        .map_err(|e| anyhow!(e.to_string()))
}
#[inline(always)]
fn adosc_batch_avx2batch(input: &AdoscInputS) -> anyhow::Result<()> {
    use my_project::indicators::adosc::AdoscBatchBuilder;
    let (h, l, c, v) = match &input.data {
        my_project::indicators::adosc::AdoscData::Candles { candles } => (
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            candles.volume.as_slice(),
        ),
        my_project::indicators::adosc::AdoscData::Slices { high, low, close, volume } =>
            (*high, *low, *close, *volume),
    };
    AdoscBatchBuilder::new()
        .kernel(Kernel::Avx2Batch)
        .apply_slices(h, l, c, v)
        .map(|_| ())
        .map_err(|e| anyhow!(e.to_string()))
}
#[inline(always)]
fn adosc_batch_avx512batch(input: &AdoscInputS) -> anyhow::Result<()> {
    use my_project::indicators::adosc::AdoscBatchBuilder;
    let (h, l, c, v) = match &input.data {
        my_project::indicators::adosc::AdoscData::Candles { candles } => (
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            candles.volume.as_slice(),
        ),
        my_project::indicators::adosc::AdoscData::Slices { high, low, close, volume } =>
            (*high, *low, *close, *volume),
    };
    AdoscBatchBuilder::new()
        .kernel(Kernel::Avx512Batch)
        .apply_slices(h, l, c, v)
        .map(|_| ())
        .map_err(|e| anyhow!(e.to_string()))
}

// AO batch wrappers (ScalarBatch/Avx2Batch/Avx512Batch)
make_batch_wrappers!(
    ao_batch,
    my_project::indicators::ao::AoBatchBuilder,
    AoInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

// OTT batch wrappers
make_batch_wrappers!(
    ott_batch, OttBatchBuilder, OttInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

// Cora Wave batch wrappers
make_batch_wrappers!(
    cora_wave_batch, CoraWaveBatchBuilder, CoraWaveInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

make_batch_wrappers!(
    wavetrend_batch, WavetrendBatchBuilder, WavetrendInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

make_batch_wrappers!(
    linearreg_intercept_batch,
    my_project::indicators::linearreg_intercept::LinearRegInterceptBatchBuilder,
    LinearRegInterceptInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

make_batch_wrappers!(
    zscore_batch, ZscoreBatchBuilder, ZscoreInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

// Custom batch wrappers for KVO (multi-series input: high/low/close/volume)
paste::paste! {
    #[inline(always)]
    fn kvo_batch_scalarbatch(_input: &KvoInputS) -> anyhow::Result<()> {
        KvoBatchBuilder::new()
            .kernel(Kernel::ScalarBatch)
            .apply_candles(&CANDLES_10K)?;
        Ok(())
    }
    #[inline(always)]
    fn kvo_batch_avx2batch(_input: &KvoInputS) -> anyhow::Result<()> {
        KvoBatchBuilder::new()
            .kernel(Kernel::Avx2Batch)
            .apply_candles(&CANDLES_10K)?;
        Ok(())
    }
    #[inline(always)]
    fn kvo_batch_avx512batch(_input: &KvoInputS) -> anyhow::Result<()> {
        KvoBatchBuilder::new()
            .kernel(Kernel::Avx512Batch)
            .apply_candles(&CANDLES_10K)?;
        Ok(())
    }
}

// Custom batch wrappers for CKSP to use candles (high/low/close)
paste::paste! {
    #[inline(always)]
    fn cksp_batch_scalarbatch(_input: &CkspInputS) -> anyhow::Result<()> {
        CkspBatchBuilder::new()
            .kernel(Kernel::ScalarBatch)
            .apply_candles(&CANDLES_10K)?;
        Ok(())
    }
    #[inline(always)]
    fn cksp_batch_avx2batch(_input: &CkspInputS) -> anyhow::Result<()> {
        CkspBatchBuilder::new()
            .kernel(Kernel::Avx2Batch)
            .apply_candles(&CANDLES_10K)?;
        Ok(())
    }
    #[inline(always)]
    fn cksp_batch_avx512batch(_input: &CkspInputS) -> anyhow::Result<()> {
        CkspBatchBuilder::new()
            .kernel(Kernel::Avx512Batch)
            .apply_candles(&CANDLES_10K)?;
        Ok(())
    }
}

impl InputLen for DeviationInputS {
    fn with_len(len: usize) -> Self {
        match len {
            10_000 => DeviationInput::with_default_candles(&*CANDLES_10K),
            100_000 => DeviationInput::with_default_candles(&*CANDLES_100K),
            1_000_000 => DeviationInput::with_default_candles(&*CANDLES_1M),
            _ => panic!("unsupported len {len}"),
        }
    }
}

// Custom batch wrappers for Squeeze Momentum to use candles (high/low/close)
paste::paste! {
    #[inline(always)]
    fn squeeze_momentum_batch_scalarbatch(_input: &SqueezeMomentumInputS) -> anyhow::Result<()> {
        SqueezeMomentumBatchBuilder::new()
            .kernel(Kernel::ScalarBatch)
            .apply_candles(&CANDLES_10K)?;
        Ok(())
    }
    #[inline(always)]
    fn squeeze_momentum_batch_avx2batch(_input: &SqueezeMomentumInputS) -> anyhow::Result<()> {
        SqueezeMomentumBatchBuilder::new()
            .kernel(Kernel::Avx2Batch)
            .apply_candles(&CANDLES_10K)?;
        Ok(())
    }
    #[inline(always)]
    fn squeeze_momentum_batch_avx512batch(_input: &SqueezeMomentumInputS) -> anyhow::Result<()> {
        SqueezeMomentumBatchBuilder::new()
            .kernel(Kernel::Avx512Batch)
            .apply_candles(&CANDLES_10K)?;
        Ok(())
    }
}


// LinearRegSlope batch wrappers
make_batch_wrappers!(
    linearreg_slope_batch, LinearRegSlopeBatchBuilder, LinearRegSlopeInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

#[inline(always)]
fn qstick_batch_scalarbatch(input: &QstickInputS) -> anyhow::Result<()> {
    let (open, close) = match &input.data {
        QstickData::Candles { candles, open_source, close_source } => (
            source_type(candles, open_source),
            source_type(candles, close_source),
        ),
        QstickData::Slices { open, close } => (*open, *close),
    };
    let sweep = QstickBatchRange { period: (5, 50, 5) };
    QstickBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .period_range(sweep.period.0, sweep.period.1, sweep.period.2)
        .apply_slices(open, close)?;
    Ok(())
}

#[inline(always)]
fn qstick_batch_avx2batch(input: &QstickInputS) -> anyhow::Result<()> {
    let (open, close) = match &input.data {
        QstickData::Candles { candles, open_source, close_source } => (
            source_type(candles, open_source),
            source_type(candles, close_source),
        ),
        QstickData::Slices { open, close } => (*open, *close),
    };
    let sweep = QstickBatchRange { period: (5, 50, 5) };
    QstickBatchBuilder::new()
        .kernel(Kernel::Avx2Batch)
        .period_range(sweep.period.0, sweep.period.1, sweep.period.2)
        .apply_slices(open, close)?;
    Ok(())
}

#[inline(always)]
fn qstick_batch_avx512batch(input: &QstickInputS) -> anyhow::Result<()> {
    let (open, close) = match &input.data {
        QstickData::Candles { candles, open_source, close_source } => (
            source_type(candles, open_source),
            source_type(candles, close_source),
        ),
        QstickData::Slices { open, close } => (*open, *close),
    };
    let sweep = QstickBatchRange { period: (5, 50, 5) };
    QstickBatchBuilder::new()
        .kernel(Kernel::Avx512Batch)
        .period_range(sweep.period.0, sweep.period.1, sweep.period.2)
        .apply_slices(open, close)?;
    Ok(())
}

// UltOsc requires OHLC; provide custom batch wrappers similar to WILLR
#[inline(always)]
fn ultosc_batch_scalarbatch(input: &UltOscInputS) -> anyhow::Result<()> {
    let (high, low, close) = match &input.data {
        UltOscData::Candles { candles, .. } => (&candles.high[..], &candles.low[..], &candles.close[..]),
        UltOscData::Slices { high, low, close } => (*high, *low, *close),
    };
    let sweep = UltOscBatchRange {
        timeperiod1: (5, 9, 2),
        timeperiod2: (12, 16, 2),
        timeperiod3: (26, 30, 2),
    };
    UltOscBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .apply_slice(high, low, close, &sweep)?;
    Ok(())
}

#[inline(always)]
fn ultosc_batch_avx2batch(input: &UltOscInputS) -> anyhow::Result<()> {
    let (high, low, close) = match &input.data {
        UltOscData::Candles { candles, .. } => (&candles.high[..], &candles.low[..], &candles.close[..]),
        UltOscData::Slices { high, low, close } => (*high, *low, *close),
    };
    let sweep = UltOscBatchRange {
        timeperiod1: (5, 9, 2),
        timeperiod2: (12, 16, 2),
        timeperiod3: (26, 30, 2),
    };
    UltOscBatchBuilder::new()
        .kernel(Kernel::Avx2Batch)
        .apply_slice(high, low, close, &sweep)?;
    Ok(())
}

#[inline(always)]
fn ultosc_batch_avx512batch(input: &UltOscInputS) -> anyhow::Result<()> {
    let (high, low, close) = match &input.data {
        UltOscData::Candles { candles, .. } => (&candles.high[..], &candles.low[..], &candles.close[..]),
        UltOscData::Slices { high, low, close } => (*high, *low, *close),
    };
    let sweep = UltOscBatchRange {
        timeperiod1: (5, 9, 2),
        timeperiod2: (12, 16, 2),
        timeperiod3: (26, 30, 2),
    };
    UltOscBatchBuilder::new()
        .kernel(Kernel::Avx512Batch)
        .apply_slice(high, low, close, &sweep)?;
    Ok(())
}

// Custom implementation for BuffAverages which requires price and volume
// Note: For simplicity, we'll just use default test data rather than trying to extract from the complex input structure
paste::paste! {
    #[inline(always)]
    fn buff_averages_batch_scalarbatch(_input: &BuffAveragesInputS) -> anyhow::Result<()> {
        // Use default test data for batch benchmarks
        let data = &CANDLES_10K.close;
        let volume = &CANDLES_10K.volume;
        BuffAveragesBatchBuilder::new()
            .kernel(Kernel::ScalarBatch)
            .apply_slices(data, volume)?;
        Ok(())
    }
    #[inline(always)]
    fn buff_averages_batch_avx2batch(_input: &BuffAveragesInputS) -> anyhow::Result<()> {
        let data = &CANDLES_10K.close;
        let volume = &CANDLES_10K.volume;
        BuffAveragesBatchBuilder::new()
            .kernel(Kernel::Avx2Batch)
            .apply_slices(data, volume)?;
        Ok(())
    }
    #[inline(always)]
    fn buff_averages_batch_avx512batch(_input: &BuffAveragesInputS) -> anyhow::Result<()> {
        let data = &CANDLES_10K.close;
        let volume = &CANDLES_10K.volume;
        BuffAveragesBatchBuilder::new()
            .kernel(Kernel::Avx512Batch)
            .apply_slices(data, volume)?;
        Ok(())
    }
}

// Custom implementation for PVI which requires close and volume
paste::paste! {
    #[inline(always)]
    fn pvi_batch_scalarbatch(_input: &PviInputS) -> anyhow::Result<()> {
        let close = &CANDLES_10K.close;
        let volume = &CANDLES_10K.volume;
        PviBatchBuilder::new()
            .kernel(Kernel::ScalarBatch)
            .apply_slices(close, volume)?;
        Ok(())
    }
    #[inline(always)]
    fn pvi_batch_avx2batch(_input: &PviInputS) -> anyhow::Result<()> {
        let close = &CANDLES_10K.close;
        let volume = &CANDLES_10K.volume;
        PviBatchBuilder::new()
            .kernel(Kernel::Avx2Batch)
            .apply_slices(close, volume)?;
        Ok(())
    }
    #[inline(always)]
    fn pvi_batch_avx512batch(_input: &PviInputS) -> anyhow::Result<()> {
        let close = &CANDLES_10K.close;
        let volume = &CANDLES_10K.volume;
        PviBatchBuilder::new()
            .kernel(Kernel::Avx512Batch)
            .apply_slices(close, volume)?;
        Ok(())
    }
}

// Custom implementation for ASO which requires full OHLC
paste::paste! {
    #[inline(always)]
    fn aso_batch_scalarbatch(input: &AsoInputS) -> anyhow::Result<()> {
        match &input.data {
            AsoData::Candles { candles, .. } => {
                AsoBatchBuilder::new()
                    .kernel(Kernel::ScalarBatch)
                    .apply_candles(candles)?;
            }
            AsoData::Slices { open, high, low, close } => {
                AsoBatchBuilder::new()
                    .kernel(Kernel::ScalarBatch)
                    .apply_slices(open, high, low, close)?;
            }
        }
        Ok(())
    }
    #[inline(always)]
    fn aso_batch_avx2batch(input: &AsoInputS) -> anyhow::Result<()> {
        match &input.data {
            AsoData::Candles { candles, .. } => {
                AsoBatchBuilder::new()
                    .kernel(Kernel::Avx2Batch)
                    .apply_candles(candles)?;
            }
            AsoData::Slices { open, high, low, close } => {
                AsoBatchBuilder::new()
                    .kernel(Kernel::Avx2Batch)
                    .apply_slices(open, high, low, close)?;
            }
        }
        Ok(())
    }
    #[inline(always)]
    fn aso_batch_avx512batch(input: &AsoInputS) -> anyhow::Result<()> {
        match &input.data {
            AsoData::Candles { candles, .. } => {
                AsoBatchBuilder::new()
                    .kernel(Kernel::Avx512Batch)
                    .apply_candles(candles)?;
            }
            AsoData::Slices { open, high, low, close } => {
                AsoBatchBuilder::new()
                    .kernel(Kernel::Avx512Batch)
                    .apply_slices(open, high, low, close)?;
            }
        }
        Ok(())
    }
}

make_batch_wrappers!(
    macz_batch, MaczBatchBuilder, MaczInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

make_batch_wrappers!(
    cwma_batch, CwmaBatchBuilder, CwmaInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

// Coppock batch variants (ScalarBatch, AVX2Batch, AVX512Batch)
make_batch_wrappers!(
    coppock_batch, my_project::indicators::coppock::CoppockBatchBuilder, CoppockInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);
bench_variants!(
    coppock_batch => CoppockInputS; None;
    coppock_batch_scalarbatch,
    coppock_batch_avx2batch,
    coppock_batch_avx512batch
);

// ADOSC single-series benches (scalar vs AVX2 vs AVX512)
bench_variants!(
    adosc => AdoscInputS; None;
    adosc_scalar,
    adosc_avx2,
    adosc_avx512
);

// ADOSC batch benches (ScalarBatch vs AVX2Batch vs AVX512Batch)
bench_variants!(
    adosc_batch => AdoscInputS; None;
    adosc_batch_scalarbatch,
    adosc_batch_avx2batch,
    adosc_batch_avx512batch
);

// BOP single-series kernel variants
bench_variants!(
    bop => BopInputS; None;
    bop_scalar,
    bop_avx2,
    bop_avx512
);

// ATR single-series benches (scalar vs AVX2 vs AVX512)
bench_variants!(
    atr => AtrInputS; None;
    atr_scalar,
    atr_avx2,
    atr_avx512
);

// ATR batch wrappers (OHLC slices)
#[inline(always)]
fn atr_batch_scalarbatch(input: &AtrInputS) -> anyhow::Result<()> {
    let (high, low, close) = match &input.data {
        my_project::indicators::atr::AtrData::Candles { candles } => (
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        ),
        my_project::indicators::atr::AtrData::Slices { high, low, close } => (*high, *low, *close),
    };
    my_project::indicators::atr::AtrBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .apply_slices(high, low, close)
        .map(|_| ())
        .map_err(|e| anyhow!(e.to_string()))
}
#[inline(always)]
fn atr_batch_avx2batch(input: &AtrInputS) -> anyhow::Result<()> {
    let (high, low, close) = match &input.data {
        my_project::indicators::atr::AtrData::Candles { candles } => (
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        ),
        my_project::indicators::atr::AtrData::Slices { high, low, close } => (*high, *low, *close),
    };
    my_project::indicators::atr::AtrBatchBuilder::new()
        .kernel(Kernel::Avx2Batch)
        .apply_slices(high, low, close)
        .map(|_| ())
        .map_err(|e| anyhow!(e.to_string()))
}
#[inline(always)]
fn atr_batch_avx512batch(input: &AtrInputS) -> anyhow::Result<()> {
    let (high, low, close) = match &input.data {
        my_project::indicators::atr::AtrData::Candles { candles } => (
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        ),
        my_project::indicators::atr::AtrData::Slices { high, low, close } => (*high, *low, *close),
    };
    my_project::indicators::atr::AtrBatchBuilder::new()
        .kernel(Kernel::Avx512Batch)
        .apply_slices(high, low, close)
        .map(|_| ())
        .map_err(|e| anyhow!(e.to_string()))
}
bench_variants!(
    atr_batch => AtrInputS; None;
    atr_batch_scalarbatch,
    atr_batch_avx2batch,
    atr_batch_avx512batch
);

make_batch_wrappers!(
    dema_batch, DemaBatchBuilder, DemaInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

// STC batch wrappers (ScalarBatch/Avx2Batch/Avx512Batch)
make_batch_wrappers!(
    stc_batch, my_project::indicators::stc::StcBatchBuilder, StcInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

make_batch_wrappers!(
    edcf_batch, EdcfBatchBuilder, EdcfInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

make_batch_wrappers!(
    ehlers_ecema_batch, EhlersEcemaBatchBuilder, EhlersEcemaInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

make_batch_wrappers!(
    ehlers_itrend_batch, EhlersITrendBatchBuilder, EhlersITrendInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

make_batch_wrappers!(
    ehlers_pma_batch, EhlersPmaBuilder, EhlersPmaInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

// StdDev batch wrappers
make_batch_wrappers!(
    stddev_batch, StdDevBatchBuilder, StdDevInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

make_batch_wrappers!(
    ehlers_kama_batch, EhlersKamaBatchBuilder, EhlersKamaInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

make_batch_wrappers!(
    ema_batch, EmaBatchBuilder, EmaInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

// ER batch variants
make_batch_wrappers!(
    er_batch, ErBatchBuilder, ErInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

make_batch_wrappers!(
    epma_batch, EpmaBatchBuilder, EpmaInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

make_batch_wrappers!(
    pma_batch, PmaBatchBuilder, PmaInputS;
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

make_batch_wrappers!(
    gaussian_batch, GaussianBatchBuilder, GaussianInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

// Pivot batch custom wrappers (needs OHLC slices)
#[inline(always)]
fn pivot_batch_scalarbatch(input: &PivotInputS) -> anyhow::Result<()> {
    let (h, l, c, o) = match input.as_ref() {
        PivotData::Candles { candles } => (
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            candles.open.as_slice(),
        ),
        PivotData::Slices { high, low, close, open } => (*high, *low, *close, *open),
    };
    PivotBatchBuilder::new().kernel(Kernel::ScalarBatch).apply_slice(h, l, c, o)?;
    Ok(())
}

// InputLen for DVDIQQE (single-series)
impl InputLen for DvdiqqeInputS {
    fn with_len(len: usize) -> Self {
        match len {
            10_000 => DvdiqqeInput::with_default_candles(&*CANDLES_10K),
            100_000 => DvdiqqeInput::with_default_candles(&*CANDLES_100K),
            1_000_000 => DvdiqqeInput::with_default_candles(&*CANDLES_1M),
            _ => panic!("unsupported len {len}"),
        }
    }
}
#[inline(always)]
fn pivot_batch_avx2batch(input: &PivotInputS) -> anyhow::Result<()> {
    let (h, l, c, o) = match input.as_ref() {
        PivotData::Candles { candles } => (
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            candles.open.as_slice(),
        ),
        PivotData::Slices { high, low, close, open } => (*high, *low, *close, *open),
    };
    PivotBatchBuilder::new().kernel(Kernel::Avx2Batch).apply_slice(h, l, c, o)?;
    Ok(())
}
#[inline(always)]
fn pivot_batch_avx512batch(input: &PivotInputS) -> anyhow::Result<()> {
    let (h, l, c, o) = match input.as_ref() {
        PivotData::Candles { candles } => (
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            candles.open.as_slice(),
        ),
        PivotData::Slices { high, low, close, open } => (*high, *low, *close, *open),
    };
    PivotBatchBuilder::new().kernel(Kernel::Avx512Batch).apply_slice(h, l, c, o)?;
    Ok(())
}

bench_variants!(
    pivot_batch => PivotInputS; None;
    pivot_batch_scalarbatch,
    pivot_batch_avx2batch,
    pivot_batch_avx512batch,
);

// Coppock single-series benches (scalar vs AVX2 vs AVX512)
bench_variants!(
    coppock => CoppockInputS; None;
    coppock_scalar,
    coppock_avx2,
    coppock_avx512
);

make_batch_wrappers!(highpass_2_pole_batch, HighPass2BatchBuilder, HighPass2InputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(highpass_batch, HighPassBatchBuilder, HighPassInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(hma_batch, HmaBatchBuilder, HmaInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(hwma_batch, HwmaBatchBuilder, HwmaInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(jma_batch, JmaBatchBuilder, JmaInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(jsa_batch, JsaBatchBuilder, JsaInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(kama_batch, KamaBatchBuilder, KamaInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(linreg_batch, LinRegBatchBuilder, LinRegInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(maaq_batch, MaaqBatchBuilder, MaaqInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(mama_batch, MamaBatchBuilder, MamaInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(mwdx_batch, MwdxBatchBuilder, MwdxInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(nma_batch, NmaBatchBuilder, NmaInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(pwma_batch, PwmaBatchBuilder, PwmaInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(reflex_batch, ReflexBatchBuilder, ReflexInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(sinwma_batch, SinWmaBatchBuilder, SinWmaInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(sma_batch, SmaBatchBuilder, SmaInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(smma_batch, SmmaBatchBuilder, SmmaInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(sqwma_batch, SqwmaBatchBuilder, SqwmaInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(srwma_batch, SrwmaBatchBuilder, SrwmaInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(supersmoother_batch, SuperSmootherBatchBuilder, SuperSmootherInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(supersmoother_3_pole_batch, SuperSmoother3PoleBatchBuilder, SuperSmoother3PoleInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(swma_batch, SwmaBatchBuilder, SwmaInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(tema_batch, TemaBatchBuilder, TemaInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(tilson_batch, TilsonBatchBuilder, TilsonInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(prb_batch, PrbBatchBuilder, PrbInputS; ScalarBatch, Avx2Batch, Avx512Batch);
// Custom batch wrappers for TRADJEMA (requires OHLC data)
#[inline(always)]
fn tradjema_batch_scalarbatch(input: &TradjemaInputS) -> anyhow::Result<()> {
    use my_project::indicators::moving_averages::tradjema::{TradjemaBatchBuilder, TradjemaData};
    use my_project::utilities::enums::Kernel;

    let (high, low, close) = match &input.data {
        TradjemaData::Candles { candles } => {
            (&candles.high[..], &candles.low[..], &candles.close[..])
        }
        TradjemaData::Slices { high, low, close } => (*high, *low, *close),
    };

    TradjemaBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .apply_slices(&high, &low, &close)?;
    Ok(())
}

#[inline(always)]
fn tradjema_batch_avx2batch(input: &TradjemaInputS) -> anyhow::Result<()> {
    use my_project::indicators::moving_averages::tradjema::{TradjemaBatchBuilder, TradjemaData};
    use my_project::utilities::enums::Kernel;

    let (high, low, close) = match &input.data {
        TradjemaData::Candles { candles } => {
            (&candles.high[..], &candles.low[..], &candles.close[..])
        }
        TradjemaData::Slices { high, low, close } => (*high, *low, *close),
    };

    TradjemaBatchBuilder::new()
        .kernel(Kernel::Avx2Batch)
        .apply_slices(&high, &low, &close)?;
    Ok(())
}

#[inline(always)]
fn tradjema_batch_avx512batch(input: &TradjemaInputS) -> anyhow::Result<()> {
    use my_project::indicators::moving_averages::tradjema::{TradjemaBatchBuilder, TradjemaData};
    use my_project::utilities::enums::Kernel;

    let (high, low, close) = match &input.data {
        TradjemaData::Candles { candles } => {
            (&candles.high[..], &candles.low[..], &candles.close[..])
        }
        TradjemaData::Slices { high, low, close } => (*high, *low, *close),
    };

    TradjemaBatchBuilder::new()
        .kernel(Kernel::Avx512Batch)
        .apply_slices(&high, &low, &close)?;
    Ok(())
}

make_batch_wrappers!(trendflex_batch, TrendFlexBatchBuilder, TrendFlexInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(trima_batch, TrimaBatchBuilder, TrimaInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(pfe_batch, PfeBatchBuilder, PfeInputS; ScalarBatch, Avx2Batch, Avx512Batch);
// UMA needs special handling for volume parameter
fn uma_batch_scalarbatch(input: &UmaInputS) -> anyhow::Result<()> {
    let slice: &[f64] = input.as_ref();
    UmaBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .apply_slice(slice, None)?;
    Ok(())
}
fn uma_batch_avx2batch(input: &UmaInputS) -> anyhow::Result<()> {
    let slice: &[f64] = input.as_ref();
    UmaBatchBuilder::new()
        .kernel(Kernel::Avx2Batch)
        .apply_slice(slice, None)?;
    Ok(())
}
fn uma_batch_avx512batch(input: &UmaInputS) -> anyhow::Result<()> {
    let slice: &[f64] = input.as_ref();
    UmaBatchBuilder::new()
        .kernel(Kernel::Avx512Batch)
        .apply_slice(slice, None)?;
    Ok(())
}

fn willr_batch_scalarbatch(input: &WillrInputS) -> anyhow::Result<()> {
    let (high, low, close) = match &input.data {
        WillrData::Candles { candles } => (&candles.high[..], &candles.low[..], &candles.close[..]),
        WillrData::Slices { high, low, close } => (*high, *low, *close),
    };

    WillrBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .apply_slices(high, low, close)?;
    Ok(())
}

fn willr_batch_avx2batch(input: &WillrInputS) -> anyhow::Result<()> {
    let (high, low, close) = match &input.data {
        WillrData::Candles { candles } => (&candles.high[..], &candles.low[..], &candles.close[..]),
        WillrData::Slices { high, low, close } => (*high, *low, *close),
    };

    WillrBatchBuilder::new()
        .kernel(Kernel::Avx2Batch)
        .apply_slices(high, low, close)?;
    Ok(())
}

fn willr_batch_avx512batch(input: &WillrInputS) -> anyhow::Result<()> {
    let (high, low, close) = match &input.data {
        WillrData::Candles { candles } => (&candles.high[..], &candles.low[..], &candles.close[..]),
        WillrData::Slices { high, low, close } => (*high, *low, *close),
    };

    WillrBatchBuilder::new()
        .kernel(Kernel::Avx512Batch)
        .apply_slices(high, low, close)?;
    Ok(())
}
make_batch_wrappers!(vidya_batch, VidyaBatchBuilder, VidyaInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(vlma_batch, VlmaBatchBuilder, VlmaInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(mom_batch, MomBatchBuilder, MomInputS; ScalarBatch, Avx2Batch, Avx512Batch);
// CVI needs OHLC (high/low) custom batch wrappers
#[inline(always)]
fn cvi_batch_scalarbatch(input: &CviInputS) -> anyhow::Result<()> {
    use my_project::utilities::enums::Kernel;
    let (high, low) = match &input.data {
        my_project::indicators::cvi::CviData::Candles(c) => (&c.high[..], &c.low[..]),
        my_project::indicators::cvi::CviData::Slices { high, low } => (*high, *low),
    };
    CviBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .apply_slices(high, low)?;
    Ok(())
}

#[inline(always)]
fn cvi_batch_avx2batch(input: &CviInputS) -> anyhow::Result<()> {
    use my_project::utilities::enums::Kernel;
    let (high, low) = match &input.data {
        my_project::indicators::cvi::CviData::Candles(c) => (&c.high[..], &c.low[..]),
        my_project::indicators::cvi::CviData::Slices { high, low } => (*high, *low),
    };
    CviBatchBuilder::new()
        .kernel(Kernel::Avx2Batch)
        .apply_slices(high, low)?;
    Ok(())
}

#[inline(always)]
fn cvi_batch_avx512batch(input: &CviInputS) -> anyhow::Result<()> {
    use my_project::utilities::enums::Kernel;
    let (high, low) = match &input.data {
        my_project::indicators::cvi::CviData::Candles(c) => (&c.high[..], &c.low[..]),
        my_project::indicators::cvi::CviData::Slices { high, low } => (*high, *low),
    };
    CviBatchBuilder::new()
        .kernel(Kernel::Avx512Batch)
        .apply_slices(high, low)?;
    Ok(())
}

// Custom implementation for VolumeAdjustedMa which requires price and volume
// Note: For simplicity, we'll just use default test data rather than trying to extract from the complex input structure
paste::paste! {
    #[inline(always)]
    fn volume_adjusted_ma_batch_scalarbatch(_input: &VolumeAdjustedMaInputS) -> anyhow::Result<()> {
        // Use default test data for batch benchmarks
        let data = &CANDLES_10K.close;
        let volume = &CANDLES_10K.volume;
        VolumeAdjustedMaBatchBuilder::new()
            .kernel(Kernel::ScalarBatch)
            .apply_slices(data, volume)?;
        Ok(())
    }
    #[inline(always)]
    fn volume_adjusted_ma_batch_avx2batch(_input: &VolumeAdjustedMaInputS) -> anyhow::Result<()> {
        let data = &CANDLES_10K.close;
        let volume = &CANDLES_10K.volume;
        VolumeAdjustedMaBatchBuilder::new()
            .kernel(Kernel::Avx2Batch)
            .apply_slices(data, volume)?;
        Ok(())
    }
    #[inline(always)]
    fn volume_adjusted_ma_batch_avx512batch(_input: &VolumeAdjustedMaInputS) -> anyhow::Result<()> {
        let data = &CANDLES_10K.close;
        let volume = &CANDLES_10K.volume;
        VolumeAdjustedMaBatchBuilder::new()
            .kernel(Kernel::Avx512Batch)
            .apply_slices(data, volume)?;
        Ok(())
    }
}

make_batch_wrappers!(vpwma_batch, VpwmaBatchBuilder, VpwmaInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(wilders_batch, WildersBatchBuilder, WildersInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(wma_batch, WmaBatchBuilder, WmaInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(zlema_batch, ZlemaBatchBuilder, ZlemaInputS; ScalarBatch, Avx2Batch, Avx512Batch);
// VI requires OHLC; provide custom batch wrappers
fn vi_batch_scalarbatch(input: &ViInputS) -> anyhow::Result<()> {
    let (high, low, close) = match &input.data {
        ViData::Candles { candles } => (&candles.high[..], &candles.low[..], &candles.close[..]),
        ViData::Slices { high, low, close } => (*high, *low, *close),
    };
    ViBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .apply_slices(high, low, close)?;
    Ok(())
}
fn vi_batch_avx2batch(input: &ViInputS) -> anyhow::Result<()> {
    let (high, low, close) = match &input.data {
        ViData::Candles { candles } => (&candles.high[..], &candles.low[..], &candles.close[..]),
        ViData::Slices { high, low, close } => (*high, *low, *close),
    };
    ViBatchBuilder::new()
        .kernel(Kernel::Avx2Batch)
        .apply_slices(high, low, close)?;
    Ok(())
}
fn vi_batch_avx512batch(input: &ViInputS) -> anyhow::Result<()> {
    let (high, low, close) = match &input.data {
        ViData::Candles { candles } => (&candles.high[..], &candles.low[..], &candles.close[..]),
        ViData::Slices { high, low, close } => (*high, *low, *close),
    };
    ViBatchBuilder::new()
        .kernel(Kernel::Avx512Batch)
        .apply_slices(high, low, close)?;
    Ok(())
}
// Keltner batch custom wrappers (needs OHLC slices + source)
#[inline(always)]
fn keltner_batch_scalarbatch(input: &KeltnerInputS) -> anyhow::Result<()> {
    let src: &[f64] = input.as_ref();
    let (h, l, c) = match &input.data {
        my_project::indicators::keltner::KeltnerData::Candles { candles, .. } => (
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        ),
        my_project::indicators::keltner::KeltnerData::Slice(h, l, c, _) => (*h, *l, *c),
    };
    my_project::indicators::keltner::KeltnerBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .apply_slice(h, l, c, src)?;
    Ok(())
}
#[inline(always)]
fn keltner_batch_avx2batch(input: &KeltnerInputS) -> anyhow::Result<()> {
    let src: &[f64] = input.as_ref();
    let (h, l, c) = match &input.data {
        my_project::indicators::keltner::KeltnerData::Candles { candles, .. } => (
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        ),
        my_project::indicators::keltner::KeltnerData::Slice(h, l, c, _) => (*h, *l, *c),
    };
    my_project::indicators::keltner::KeltnerBatchBuilder::new()
        .kernel(Kernel::Avx2Batch)
        .apply_slice(h, l, c, src)?;
    Ok(())
}
#[inline(always)]
fn keltner_batch_avx512batch(input: &KeltnerInputS) -> anyhow::Result<()> {
    let src: &[f64] = input.as_ref();
    let (h, l, c) = match &input.data {
        my_project::indicators::keltner::KeltnerData::Candles { candles, .. } => (
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        ),
        my_project::indicators::keltner::KeltnerData::Slice(h, l, c, _) => (*h, *l, *c),
    };
    my_project::indicators::keltner::KeltnerBatchBuilder::new()
        .kernel(Kernel::Avx512Batch)
        .apply_slice(h, l, c, src)?;
    Ok(())
}
make_batch_wrappers!(trix_batch, TrixBatchBuilder, TrixInputS; ScalarBatch, Avx2Batch, Avx512Batch);
// ChandelierExit needs special handling for apply_slices
fn chandelier_exit_batch_scalarbatch(input: &ChandelierExitInputS) -> anyhow::Result<()> {
    // ChandelierExit requires high, low, and close data
    // For benchmarking, we'll use the same data for all three
    let slice: &[f64] = input.as_ref();
    CeBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .apply_slices(slice, slice, slice)?;
    Ok(())
}
fn chandelier_exit_batch_avx2batch(input: &ChandelierExitInputS) -> anyhow::Result<()> {
    let slice: &[f64] = input.as_ref();
    CeBatchBuilder::new()
        .kernel(Kernel::Avx2Batch)
        .apply_slices(slice, slice, slice)?;
    Ok(())
}
fn chandelier_exit_batch_avx512batch(input: &ChandelierExitInputS) -> anyhow::Result<()> {
    let slice: &[f64] = input.as_ref();
    CeBatchBuilder::new()
        .kernel(Kernel::Avx512Batch)
        .apply_slices(slice, slice, slice)?;
    Ok(())
}
// PercentileNearestRank needs special handling for apply method
fn percentile_nearest_rank_batch_scalarbatch(
    input: &PercentileNearestRankInputS,
) -> anyhow::Result<()> {
    let slice: &[f64] = input.as_ref();
    PercentileNearestRankBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .apply(slice)?;
    Ok(())
}
fn percentile_nearest_rank_batch_avx2batch(
    input: &PercentileNearestRankInputS,
) -> anyhow::Result<()> {
    let slice: &[f64] = input.as_ref();
    PercentileNearestRankBatchBuilder::new()
        .kernel(Kernel::Avx2Batch)
        .apply(slice)?;
    Ok(())
}
fn percentile_nearest_rank_batch_avx512batch(
    input: &PercentileNearestRankInputS,
) -> anyhow::Result<()> {
    let slice: &[f64] = input.as_ref();
    PercentileNearestRankBatchBuilder::new()
        .kernel(Kernel::Avx512Batch)
        .apply(slice)?;
    Ok(())
}

make_batch_wrappers!(otto_batch, OttoBatchBuilder, OttoInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(var_batch, VarBatchBuilder, VarInputS; ScalarBatch, Avx2Batch, Avx512Batch);

// NATR (OHLC batch) wrappers
#[inline(always)]
fn natr_batch_scalarbatch(input: &NatrInputS) -> anyhow::Result<()> {
    let (high, low, close) = match &input.data {
        NatrData::Candles { candles } => (&candles.high[..], &candles.low[..], &candles.close[..]),
        NatrData::Slices { high, low, close } => (*high, *low, *close),
    };
    NatrBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .apply_slices(high, low, close)?;
    Ok(())
}

// Donchian batch custom wrappers (high/low inputs)
#[inline(always)]
fn donchian_batch_scalarbatch(input: &DonchianInputS) -> anyhow::Result<()> {
    use my_project::indicators::donchian::DonchianBatchBuilder;
    let (high, low) = match &input.data {
        my_project::indicators::donchian::DonchianData::Candles { candles } => {
            (&candles.high[..], &candles.low[..])
        }
        my_project::indicators::donchian::DonchianData::Slices { high, low } => (*high, *low),
    };
    DonchianBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .apply_slices(high, low)?;
    Ok(())
}

#[inline(always)]
fn donchian_batch_avx2batch(input: &DonchianInputS) -> anyhow::Result<()> {
    use my_project::indicators::donchian::DonchianBatchBuilder;
    let (high, low) = match &input.data {
        my_project::indicators::donchian::DonchianData::Candles { candles } => {
            (&candles.high[..], &candles.low[..])
        }
        my_project::indicators::donchian::DonchianData::Slices { high, low } => (*high, *low),
    };
    DonchianBatchBuilder::new()
        .kernel(Kernel::Avx2Batch)
        .apply_slices(high, low)?;
    Ok(())
}

#[inline(always)]
fn donchian_batch_avx512batch(input: &DonchianInputS) -> anyhow::Result<()> {
    use my_project::indicators::donchian::DonchianBatchBuilder;
    let (high, low) = match &input.data {
        my_project::indicators::donchian::DonchianData::Candles { candles } => {
            (&candles.high[..], &candles.low[..])
        }
        my_project::indicators::donchian::DonchianData::Slices { high, low } => (*high, *low),
    };
    DonchianBatchBuilder::new()
        .kernel(Kernel::Avx512Batch)
        .apply_slices(high, low)?;
    Ok(())
}

#[inline(always)]
fn natr_batch_avx2batch(input: &NatrInputS) -> anyhow::Result<()> {
    let (high, low, close) = match &input.data {
        NatrData::Candles { candles } => (&candles.high[..], &candles.low[..], &candles.close[..]),
        NatrData::Slices { high, low, close } => (*high, *low, *close),
    };
    NatrBatchBuilder::new()
        .kernel(Kernel::Avx2Batch)
        .apply_slices(high, low, close)?;
    Ok(())
}

#[inline(always)]
fn natr_batch_avx512batch(input: &NatrInputS) -> anyhow::Result<()> {
    let (high, low, close) = match &input.data {
        NatrData::Candles { candles } => (&candles.high[..], &candles.low[..], &candles.close[..]),
        NatrData::Slices { high, low, close } => (*high, *low, *close),
    };
    NatrBatchBuilder::new()
        .kernel(Kernel::Avx512Batch)
        .apply_slices(high, low, close)?;
    Ok(())
}

// Other indicators batch wrappers
// Custom implementation for AVSL which requires close, low, and volume
// Note: For simplicity, we'll just use default test data rather than trying to extract from the complex input structure
paste::paste! {
    #[inline(always)]
    fn avsl_batch_scalarbatch(_input: &AvslInputS) -> anyhow::Result<()> {
        // Use default test data for batch benchmarks
        let close = &CANDLES_10K.close;
        let low = &CANDLES_10K.low;
        let volume = &CANDLES_10K.volume;
        AvslBatchBuilder::new()
            .kernel(Kernel::ScalarBatch)
            .apply_slices(close, low, volume)?;
        Ok(())
    }
    #[inline(always)]
    fn avsl_batch_avx2batch(_input: &AvslInputS) -> anyhow::Result<()> {
        let close = &CANDLES_10K.close;
        let low = &CANDLES_10K.low;
        let volume = &CANDLES_10K.volume;
        AvslBatchBuilder::new()
            .kernel(Kernel::Avx2Batch)
            .apply_slices(close, low, volume)?;
        Ok(())
    }
    #[inline(always)]
    fn avsl_batch_avx512batch(_input: &AvslInputS) -> anyhow::Result<()> {
        let close = &CANDLES_10K.close;
        let low = &CANDLES_10K.low;
        let volume = &CANDLES_10K.volume;
        AvslBatchBuilder::new()
            .kernel(Kernel::Avx512Batch)
            .apply_slices(close, low, volume)?;
        Ok(())
    }
}
make_batch_wrappers!(dma_batch, DmaBatchBuilder, DmaInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(ehma_batch, EhmaBatchBuilder, EhmaInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(range_filter_batch, RangeFilterBatchBuilder, RangeFilterInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(sama_batch, SamaBatchBuilder, SamaInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(wto_batch, WtoBatchBuilder, WtoInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(nama_batch, NamaBatchBuilder, NamaInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(net_myrsi_batch, NetMyrsiBatchBuilder, NetMyrsiInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(cci_cycle_batch, CciCycleBatchBuilder, CciCycleInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(cmo_batch, CmoBatchBuilder, CmoInputS; ScalarBatch, Avx2Batch, Avx512Batch);
// TODO: FvgTrailingStopInput uses apply_slices() not apply_slice() - needs custom batch wrapper
// make_batch_wrappers!(fvg_trailing_stop_batch, FvgTsBatchBuilder, FvgTrailingStopInputS; ScalarBatch, Avx2Batch, Avx512Batch);
// TODO: HalfTrendInput doesn't implement AsRef - needs custom batch wrapper
// make_batch_wrappers!(halftrend_batch, HalfTrendBatchBuilder, HalfTrendInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(reverse_rsi_batch, ReverseRsiBatchBuilder, ReverseRsiInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(vama_batch, VamaBatchBuilder, VamaInputS; ScalarBatch, Avx2Batch, Avx512Batch);
make_batch_wrappers!(decycler_batch, my_project::indicators::decycler::DecyclerBatchBuilder, DecyclerInputS; ScalarBatch, Avx2Batch, Avx512Batch);

bench_variants!(
    bandpass => BandPassInputS; None;
    bandpass_scalar,
    bandpass_avx2,
    bandpass_avx512
);
make_batch_wrappers!(deviation_batch, DeviationBatchBuilder, DeviationInputS; ScalarBatch, Avx2Batch, Avx512Batch);
// Correlation Cycle batch wrappers
make_batch_wrappers!(
    correlation_cycle_batch, CorrelationCycleBatchBuilder, CorrelationCycleInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

// Bollinger Bands Width batch kernel variants
make_batch_wrappers!(
    bollinger_bands_width_batch,
    BollingerBandsWidthBatchBuilder,
    BollingerBandsWidthInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);
make_batch_wrappers!(srsi_batch, SrsiBatchBuilder, SrsiInputS; ScalarBatch, Avx2Batch, Avx512Batch);
// Damiani Volatmeter batch variants (ScalarBatch/Avx2Batch/Avx512Batch)
make_batch_wrappers!(
    damiani_volatmeter_batch,
    my_project::indicators::damiani_volatmeter::DamianiVolatmeterBatchBuilder,
    DamianiVolatmeterInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);
make_batch_wrappers!(kst_batch, KstBatchBuilder, KstInputS; ScalarBatch, Avx2Batch, Avx512Batch);

bench_variants!(
    alma_batch => AlmaInputS; Some(232);
    alma_batch_scalarbatch,
    alma_batch_avx2batch,
    alma_batch_avx512batch
);

// AO single-series variants (scalar vs AVX2 vs AVX512)
bench_variants!(
    ao => AoInputS; None;
    ao_scalar,
    ao_avx2,
    ao_avx512
);

// AO batch variants (ScalarBatch/AVX2Batch/AVX512Batch)
bench_variants!(
    ao_batch => AoInputS; None;
    ao_batch_scalarbatch,
    ao_batch_avx2batch,
    ao_batch_avx512batch
);

bench_variants!(
    damiani_volatmeter_batch => DamianiVolatmeterInputS; None;
    damiani_volatmeter_batch_scalarbatch,
    damiani_volatmeter_batch_avx2batch,
    damiani_volatmeter_batch_avx512batch
);

// MACD single-series: compare scalar vs AVX2 vs AVX512
bench_variants!(
    macd => MacdInputS; None;
    macd_scalar,
    macd_avx2,
    macd_avx512,
);

// Damiani Volatmeter single-series variants (Scalar/AVX2/AVX512)
bench_variants!(
    damiani_volatmeter => DamianiVolatmeterInputS; None;
    damiani_volatmeter_scalar,
    damiani_volatmeter_avx2,
    damiani_volatmeter_avx512,
);

// ER single-series: compare scalar vs AVX2 vs AVX512
bench_variants!(
    er => ErInputS; None;
    er_scalar,
    er_avx2,
    er_avx512,
);

// MACD batch variants (ScalarBatch/Avx2Batch/Avx512Batch)
make_batch_wrappers!(
    macd_batch, my_project::indicators::macd::MacdBatchBuilder, MacdInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);
bench_variants!(
    macd_batch => MacdInputS; None;
    macd_batch_scalarbatch,
    macd_batch_avx2batch,
    macd_batch_avx512batch
);
bench_variants!(
    adx_batch => AdxInputS; Some(14);
    adx_batch_scalarbatch,
    adx_batch_avx2batch,
    adx_batch_avx512batch,
);

bench_variants!(
    buff_averages_batch => BuffAveragesInputS; None;
    buff_averages_batch_scalarbatch,
    buff_averages_batch_avx2batch,
    buff_averages_batch_avx512batch
);

make_batch_wrappers!(bandpass_batch, BandPassBatchBuilder, BandPassInputS; ScalarBatch, Avx2Batch, Avx512Batch);
bench_variants!(
    bandpass_batch => BandPassInputS; None;
    bandpass_batch_scalarbatch,
    bandpass_batch_avx2batch,
    bandpass_batch_avx512batch
);

bench_variants!(
    decycler_batch => DecyclerInputS; None;
    decycler_batch_scalarbatch,
    decycler_batch_avx2batch,
    decycler_batch_avx512batch
);

bench_variants!(
    linearreg_angle => LinearregAngleInputS; Some(14);
    linearreg_angle_scalar,
    linearreg_angle_avx2,
    linearreg_angle_avx512,
);

bench_variants!(
    cmo => CmoInputS; Some(14);
    cmo_scalar,
    cmo_avx2,
    cmo_avx512,
);

bench_variants!(
    zscore_batch => ZscoreInputS; Some(14);
    zscore_batch_scalarbatch,
    zscore_batch_avx2batch,
    zscore_batch_avx512batch
);

// SAR single-series kernel variants (Scalar/Avx2/Avx512)
bench_variants!(
    sar => SarInputS; None;
    sar_scalar,
    sar_avx2,
    sar_avx512
);

// CORREL_HL single-series kernel variants (Scalar/Avx2/Avx512)
bench_variants!(
    correl_hl => CorrelHlInputS; None;
    correl_hl_scalar,
    correl_hl_avx2,
    correl_hl_avx512
);

bench_variants!(
    natr => NatrInputS; None;
    natr_scalar,
    natr_avx2,
    natr_avx512
);

// EFI single-series kernel variants (Scalar/Avx2/Avx512)
bench_variants!(
    efi => EfiInputS; None;
    efi_scalar,
    efi_avx2,
    efi_avx512
);

bench_variants!(
    fisher_batch => FisherInputS; None;
    fisher_batch_scalarbatch,
    fisher_batch_avx2batch,
    fisher_batch_avx512batch
);

bench_variants!(
    marketfi => MarketefiInputS; None;
    marketfi_scalar,
    marketfi_avx2,
    marketfi_avx512
);

bench_variants!(
    var_batch => VarInputS; Some(14);
    var_batch_scalarbatch,
    var_batch_avx2batch,
    var_batch_avx512batch
);

bench_variants!(
    correlation_cycle => CorrelationCycleInputS; None;
    correlation_cycle_scalar,
    correlation_cycle_avx2,
    correlation_cycle_avx512,
);

bench_variants!(
    bollinger_bands_width => BollingerBandsWidthInputS; None;
    bollinger_bands_width_scalar,
    bollinger_bands_width_avx2,
    bollinger_bands_width_avx512
);

bench_variants!(
    correlation_cycle_batch => CorrelationCycleInputS; None;
    correlation_cycle_batch_scalarbatch,
    correlation_cycle_batch_avx2batch,
    correlation_cycle_batch_avx512batch
);

bench_variants!(
    bollinger_bands_width_batch => BollingerBandsWidthInputS; None;
    bollinger_bands_width_batch_scalarbatch,
    bollinger_bands_width_batch_avx2batch,
    bollinger_bands_width_batch_avx512batch
);

bench_variants!(
    kvo_batch => KvoInputS; Some(200);
    kvo_batch_scalarbatch,
    kvo_batch_avx2batch,
    kvo_batch_avx512batch
);

bench_variants!(
    squeeze_momentum_batch => SqueezeMomentumInputS; Some(3);
    squeeze_momentum_batch_scalarbatch,
    squeeze_momentum_batch_avx2batch,
    squeeze_momentum_batch_avx512batch
);

bench_variants!(
    pvi => PviInputS; None;
    pvi_scalar,
    pvi_avx2,
    pvi_avx512,
);

bench_variants!(
    dti => DtiInputS; None;
    dti_scalar,
    dti_avx2,
    dti_avx512,
);

bench_variants!(
    pvi_batch => PviInputS; Some(227);
    pvi_batch_scalarbatch,
    pvi_batch_avx2batch,
    pvi_batch_avx512batch,
);

bench_variants!(
    er_batch => ErInputS; None;
    er_batch_scalarbatch,
    er_batch_avx2batch,
    er_batch_avx512batch
);

// LinearRegSlope batch variants (advisory unless row-specific kernels are optimized)
bench_variants!(
    linearreg_slope_batch => LinearRegSlopeInputS; Some(14);
    linearreg_slope_batch_scalarbatch,
    linearreg_slope_batch_avx2batch,
    linearreg_slope_batch_avx512batch
);

// ROCR scalar-vs-SIMD single-series variants
bench_variants!(
    rocr => RocrInputS; None;
    rocr_scalar,
    rocr_avx2,
    rocr_avx512
);

// DM scalar-vs-SIMD single-series variants
bench_variants!(
    dm => DmInputS; None;
    dm_scalar,
    dm_avx2,
    dm_avx512
);

// DM batch wrappers (two-slice input)
#[inline(always)]
fn dm_batch_scalarbatch(input: &DmInputS) -> anyhow::Result<()> {
    let (high, low) = match &input.data {
        DmData::Candles { candles } => (&candles.high[..], &candles.low[..]),
        DmData::Slices { high, low } => (*high, *low),
    };
    DmBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .apply_slices(high, low)
        .map(|_| ())
        .map_err(|e| anyhow!(e.to_string()))
}

#[inline(always)]
fn dm_batch_avx2batch(input: &DmInputS) -> anyhow::Result<()> {
    let (high, low) = match &input.data {
        DmData::Candles { candles } => (&candles.high[..], &candles.low[..]),
        DmData::Slices { high, low } => (*high, *low),
    };
    DmBatchBuilder::new()
        .kernel(Kernel::Avx2Batch)
        .apply_slices(high, low)
        .map(|_| ())
        .map_err(|e| anyhow!(e.to_string()))
}

#[inline(always)]
fn dm_batch_avx512batch(input: &DmInputS) -> anyhow::Result<()> {
    let (high, low) = match &input.data {
        DmData::Candles { candles } => (&candles.high[..], &candles.low[..]),
        DmData::Slices { high, low } => (*high, *low),
    };
    DmBatchBuilder::new()
        .kernel(Kernel::Avx512Batch)
        .apply_slices(high, low)
        .map(|_| ())
        .map_err(|e| anyhow!(e.to_string()))
}

bench_variants!(
    dm_batch => DmInputS; Some(14);
    dm_batch_scalarbatch,
    dm_batch_avx2batch,
    dm_batch_avx512batch,
);

// LinearRegSlope scalar-vs-SIMD single-series variants
bench_variants!(
    linearreg_slope => LinearRegSlopeInputS; Some(14);
    linearreg_slope_scalar,
    linearreg_slope_avx2,
    linearreg_slope_avx512
);

// OTT scalar-vs-SIMD single-series variants
bench_variants!(
    ott => OttInputS; None;
    ott_scalar,
    ott_avx2,
    ott_avx512
);

bench_variants!(
    stddev_batch => StdDevInputS; Some(5);
    stddev_batch_scalarbatch,
    stddev_batch_avx2batch,
    stddev_batch_avx512batch
);

bench_variants!(
    macz_batch => MaczInputS; Some(232);
    macz_batch_scalarbatch,
    macz_batch_avx2batch,
    macz_batch_avx512batch
);

// OTT batch variants (advisory unless row-specific kernels are optimized)
bench_variants!(
    ott_batch => OttInputS; None;
    ott_batch_scalarbatch,
    ott_batch_avx2batch,
    ott_batch_avx512batch
);

// CoRa Wave batch variants (advisory unless row-specific SIMD is optimized)
bench_variants!(
    cora_wave_batch => CoraWaveInputS; None;
    cora_wave_batch_scalarbatch,
    cora_wave_batch_avx2batch,
    cora_wave_batch_avx512batch,
);

bench_variants!(
    kst_batch => KstInputS; None;
    kst_batch_scalarbatch,
    kst_batch_avx2batch,
    kst_batch_avx512batch
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
    sar_batch => SarInputS; None;
    sar_batch_scalarbatch,
    sar_batch_avx2batch,
    sar_batch_avx512batch
);

bench_variants!(
    natr_batch => NatrInputS; None;
    natr_batch_scalarbatch,
    natr_batch_avx2batch,
    natr_batch_avx512batch
);

bench_variants!(
    ehlers_ecema => EhlersEcemaInputS; None;
    ehlers_ecema_scalar,
    ehlers_ecema_avx2,
    ehlers_ecema_avx512,
);

bench_variants!(
    ehlers_ecema_batch => EhlersEcemaInputS; Some(227);
    ehlers_ecema_batch_scalarbatch,
    ehlers_ecema_batch_avx2batch,
    ehlers_ecema_batch_avx512batch,
);

bench_variants!(
    ehlers_itrend_batch => EhlersITrendInputS; Some(227);
    ehlers_itrend_batch_scalarbatch,
    ehlers_itrend_batch_avx2batch,
    ehlers_itrend_batch_avx512batch,
);

bench_variants!(
    ehlers_pma_batch => EhlersPmaInputS; Some(227);
    ehlers_pma_batch_scalarbatch,
    ehlers_pma_batch_avx2batch,
    ehlers_pma_batch_avx512batch,
);

bench_variants!(
    ehlers_kama_batch => EhlersKamaInputS; Some(20);
    ehlers_kama_batch_scalarbatch,
    ehlers_kama_batch_avx2batch,
    ehlers_kama_batch_avx512batch,
);

bench_variants!(
    ema_batch => EmaInputS; Some(227);
    ema_batch_scalarbatch,
    ema_batch_avx2batch,
    ema_batch_avx512batch,
);

bench_variants!(
    stc_batch => StcInputS; None;
    stc_batch_scalarbatch,
    stc_batch_avx2batch,
    stc_batch_avx512batch,
);

bench_variants!(
    epma_batch => EpmaInputS; Some(227);
    epma_batch_scalarbatch,
    epma_batch_avx2batch,
    epma_batch_avx512batch,
);

bench_variants!(
    pma_batch => PmaInputS; Some(227);
    pma_batch_scalarbatch,
    pma_batch_avx2batch,
    pma_batch_avx512batch,
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
    gaussian_batch => GaussianInputS; Some(227);
    gaussian_batch_scalarbatch,
    gaussian_batch_avx2batch,
    gaussian_batch_avx512batch,
);

bench_variants!(
    highpass_2_pole_batch => HighPass2InputS; Some(227);
    highpass_2_pole_batch_scalarbatch,
    highpass_2_pole_batch_avx2batch,
    highpass_2_pole_batch_avx512batch,
);

bench_variants!(
    apo => ApoInputS; None;
    apo_scalar,
    apo_avx2,
    apo_avx512,
);

bench_variants!(
    highpass_batch => HighPassInputS; Some(227);
    highpass_batch_scalarbatch,
    highpass_batch_avx2batch,
    highpass_batch_avx512batch,
);

bench_variants!(
    hma_batch => HmaInputS; Some(227);
    hma_batch_scalarbatch,
    hma_batch_avx2batch,
    hma_batch_avx512batch,
);

// STC single-series scalar vs AVX benches
bench_variants!(
    stc => StcInputS; None;
    stc_scalar,
    stc_avx2,
    stc_avx512,
);

// ROCP single-series scalar vs AVX benches
bench_variants!(
    rocp => RocpInputS; None;
    rocp_scalar,
    rocp_avx2,
    rocp_avx512,
);

bench_variants!(
    hwma_batch => HwmaInputS; Some(227);
    hwma_batch_scalarbatch,
    hwma_batch_avx2batch,
    hwma_batch_avx512batch,
);

bench_variants!(
    jma_batch => JmaInputS; Some(227);
    jma_batch_scalarbatch,
    jma_batch_avx2batch,
    jma_batch_avx512batch,
);

bench_variants!(
    jsa_batch => JsaInputS; Some(227);
    jsa_batch_scalarbatch,
    jsa_batch_avx2batch,
    jsa_batch_avx512batch,
);

bench_variants!(
    kama_batch => KamaInputS; Some(227);
    kama_batch_scalarbatch,
    kama_batch_avx2batch,
    kama_batch_avx512batch,
);

bench_variants!(
    linreg_batch => LinRegInputS; Some(227);
    linreg_batch_scalarbatch,
    linreg_batch_avx2batch,
    linreg_batch_avx512batch,
);

bench_variants!(
    linearreg_intercept => LinearRegInterceptInputS; None;
    linearreg_intercept_scalar,
    linearreg_intercept_avx2,
    linearreg_intercept_avx512,
);

bench_variants!(
    linearreg_intercept_batch => LinearRegInterceptInputS; Some(227);
    linearreg_intercept_batch_scalarbatch,
    linearreg_intercept_batch_avx2batch,
    linearreg_intercept_batch_avx512batch,
);

bench_variants!(
    maaq_batch => MaaqInputS; Some(227);
    maaq_batch_scalarbatch,
    maaq_batch_avx2batch,
    maaq_batch_avx512batch,
);

bench_variants!(
    mama_batch => MamaInputS; Some(227);
    mama_batch_scalarbatch,
    mama_batch_avx2batch,
    mama_batch_avx512batch,
);

bench_variants!(
    mwdx_batch => MwdxInputS; Some(227);
    mwdx_batch_scalarbatch,
    mwdx_batch_avx2batch,
    mwdx_batch_avx512batch,
);

bench_variants!(
    nma_batch => NmaInputS; Some(227);
    nma_batch_scalarbatch,
    nma_batch_avx2batch,
    nma_batch_avx512batch,
);


bench_variants!(
    pwma_batch => PwmaInputS; Some(227);
    pwma_batch_scalarbatch,
    pwma_batch_avx2batch,
    pwma_batch_avx512batch,
);

bench_variants!(
    reflex_batch => ReflexInputS; Some(227);
    reflex_batch_scalarbatch,
    reflex_batch_avx2batch,
    reflex_batch_avx512batch,
);

bench_variants!(
    sinwma_batch => SinWmaInputS; Some(227);
    sinwma_batch_scalarbatch,
    sinwma_batch_avx2batch,
    sinwma_batch_avx512batch,
);

bench_variants!(
    sma_batch => SmaInputS; Some(227);
    sma_batch_scalarbatch,
    sma_batch_avx2batch,
    sma_batch_avx512batch,
);

bench_variants!(
    smma_batch => SmmaInputS; Some(227);
    smma_batch_scalarbatch,
    smma_batch_avx2batch,
    smma_batch_avx512batch,
);

bench_variants!(
    sqwma_batch => SqwmaInputS; Some(227);
    sqwma_batch_scalarbatch,
    sqwma_batch_avx2batch,
    sqwma_batch_avx512batch,
);

bench_variants!(
    srwma_batch => SrwmaInputS; Some(227);
    srwma_batch_scalarbatch,
    srwma_batch_avx2batch,
    srwma_batch_avx512batch,
);

bench_variants!(
    supersmoother_batch => SuperSmootherInputS; Some(227);
    supersmoother_batch_scalarbatch,
    supersmoother_batch_avx2batch,
    supersmoother_batch_avx512batch,
);

bench_variants!(
    supersmoother_3_pole_batch => SuperSmoother3PoleInputS; Some(227);
    supersmoother_3_pole_batch_scalarbatch,
    supersmoother_3_pole_batch_avx2batch,
    supersmoother_3_pole_batch_avx512batch,
);

bench_variants!(
    swma_batch => SwmaInputS; Some(227);
    swma_batch_scalarbatch,
    swma_batch_avx2batch,
    swma_batch_avx512batch,
);

bench_variants!(
    tema_batch => TemaInputS; Some(227);
    tema_batch_scalarbatch,
    tema_batch_avx2batch,
    tema_batch_avx512batch,
);

bench_variants!(
    tilson_batch => TilsonInputS; Some(227);
    tilson_batch_scalarbatch,
    tilson_batch_avx2batch,
    tilson_batch_avx512batch,
);

bench_variants!(
    tradjema_batch => TradjemaInputS; Some(227);
    tradjema_batch_scalarbatch,
    tradjema_batch_avx2batch,
    tradjema_batch_avx512batch,
);

bench_variants!(
    trendflex_batch => TrendFlexInputS; Some(227);
    trendflex_batch_scalarbatch,
    trendflex_batch_avx2batch,
    trendflex_batch_avx512batch,
);

bench_variants!(
    trima_batch => TrimaInputS; Some(227);
    trima_batch_scalarbatch,
    trima_batch_avx2batch,
    trima_batch_avx512batch,
);

bench_variants!(
    cfo => CfoInputS; Some(14);
    cfo_scalar,
    cfo_avx2,
    cfo_avx512,
);

make_batch_wrappers!(
    cfo_batch, CfoBatchBuilder, CfoInputS;
    ScalarBatch, Avx2Batch, Avx512Batch
);

bench_variants!(
    cfo_batch => CfoInputS; Some(14);
    cfo_batch_scalarbatch,
    cfo_batch_avx2batch,
    cfo_batch_avx512batch,
);

bench_variants!(
    uma_batch => UmaInputS; Some(227);
    uma_batch_scalarbatch,
    uma_batch_avx2batch,
    uma_batch_avx512batch,
);

bench_variants!(
    vidya_batch => VidyaInputS; Some(227);
    vidya_batch_scalarbatch,
    vidya_batch_avx2batch,
    vidya_batch_avx512batch,
);

bench_variants!(
    vlma_batch => VlmaInputS; Some(227);
    vlma_batch_scalarbatch,
    vlma_batch_avx2batch,
    vlma_batch_avx512batch,
);

bench_variants!(
    volume_adjusted_ma_batch => VolumeAdjustedMaInputS; None;
    volume_adjusted_ma_batch_scalarbatch,
    volume_adjusted_ma_batch_avx2batch,
    volume_adjusted_ma_batch_avx512batch,
);

bench_variants!(
    vpwma_batch => VpwmaInputS; Some(227);
    vpwma_batch_scalarbatch,
    vpwma_batch_avx2batch,
    vpwma_batch_avx512batch,
);

bench_variants!(
    vi_batch => ViInputS; Some(227);
    vi_batch_scalarbatch,
    vi_batch_avx2batch,
    vi_batch_avx512batch,
);

bench_variants!(
    willr_batch => WillrInputS; Some(227);
    willr_batch_scalarbatch,
    willr_batch_avx2batch,
    willr_batch_avx512batch,
);

bench_variants!(
    wilders_batch => WildersInputS; Some(227);
    wilders_batch_scalarbatch,
    wilders_batch_avx2batch,
    wilders_batch_avx512batch,
);

bench_variants!(
    wma_batch => WmaInputS; Some(227);
    wma_batch_scalarbatch,
    wma_batch_avx2batch,
    wma_batch_avx512batch,
);

bench_variants!(
    mom_batch => MomInputS; None;
    mom_batch_scalarbatch,
    mom_batch_avx2batch,
    mom_batch_avx512batch,
);

bench_variants!(
    zlema_batch => ZlemaInputS; Some(227);
    zlema_batch_scalarbatch,
    zlema_batch_avx2batch,
    zlema_batch_avx512batch,
);

bench_variants!(
    keltner_batch => KeltnerInputS; Some(227);
    keltner_batch_scalarbatch,
    keltner_batch_avx2batch,
    keltner_batch_avx512batch,
);

// TRIX single-series and batch variants
bench_variants!(
    trix => TrixInputS; None;
    trix_scalar,
    trix_avx2,
    trix_avx512,
);
// OBV single-series variants (scalar vs AVX2/AVX512)
bench_variants!(
    obv => ObvInputS; None;
    obv_scalar,
    obv_avx2,
    obv_avx512,
);
bench_variants!(
    trix_batch => TrixInputS; None;
    trix_batch_scalarbatch,
    trix_batch_avx2batch,
    trix_batch_avx512batch,
);

bench_variants!(
    chandelier_exit_batch => ChandelierExitInputS; Some(227);
    chandelier_exit_batch_scalarbatch,
    chandelier_exit_batch_avx2batch,
    chandelier_exit_batch_avx512batch,
);

bench_variants!(
    otto_batch => OttoInputS; Some(227);
    otto_batch_scalarbatch,
    otto_batch_avx2batch,
    otto_batch_avx512batch,
);

bench_variants!(
    percentile_nearest_rank_batch => PercentileNearestRankInputS; Some(227);
    percentile_nearest_rank_batch_scalarbatch,
    percentile_nearest_rank_batch_avx2batch,
    percentile_nearest_rank_batch_avx512batch,
);

// Other indicators batch variants
bench_variants!(
    avsl_batch => AvslInputS; Some(200);
    avsl_batch_scalarbatch,
    avsl_batch_avx2batch,
    avsl_batch_avx512batch,
);

bench_variants!(
    dma_batch => DmaInputS; Some(200);
    dma_batch_scalarbatch,
    dma_batch_avx2batch,
    dma_batch_avx512batch,
);

bench_variants!(
    range_filter_batch => RangeFilterInputS; Some(200);
    range_filter_batch_scalarbatch,
    range_filter_batch_avx2batch,
    range_filter_batch_avx512batch,
);

bench_variants!(
    ehma_batch => EhmaInputS; Some(200);
    ehma_batch_scalarbatch,
    ehma_batch_avx2batch,
    ehma_batch_avx512batch,
);

bench_variants!(
    sama_batch => SamaInputS; Some(200);
    sama_batch_scalarbatch,
    sama_batch_avx2batch,
    sama_batch_avx512batch,
);

bench_variants!(
    wto_batch => WtoInputS; Some(200);
    wto_batch_scalarbatch,
    wto_batch_avx2batch,
    wto_batch_avx512batch,
);

bench_variants!(
    nama_batch => NamaInputS; Some(30);
    nama_batch_scalarbatch,
    nama_batch_avx2batch,
    nama_batch_avx512batch,
);

bench_variants!(
    alma => AlmaInputS; None;
    alma_scalar,
    alma_avx2,
    alma_avx512,
);

bench_variants!(
    adx => AdxInputS; Some(14);
    adx_scalar,
    adx_avx2,
    adx_avx512,
);

bench_variants!(
    nadaraya_watson_envelope => NweInputS; None;
    nadaraya_watson_envelope_scalar,
    nadaraya_watson_envelope_avx2,
    nadaraya_watson_envelope_avx512,
);

bench_variants!(
    buff_averages => BuffAveragesInputS; None;
    buff_averages_scalar,
    buff_averages_avx2,
    buff_averages_avx512,
);

bench_variants!(
    eri => EriInputS; None;
    eri_scalar,
    eri_avx2,
    eri_avx512,
);

bench_variants!(
    zscore => ZscoreInputS; Some(14);
    zscore_scalar,
    zscore_avx2,
    zscore_avx512,
);

bench_variants!(
    var => VarInputS; Some(14);
    var_scalar,
    var_avx2,
    var_avx512,
);

bench_variants!(
    mab => MabInputS; None;
    mab_scalar,
    mab_avx2,
    mab_avx512,
);

bench_variants!(
    macz => MaczInputS; None;
    macz_scalar,
    macz_avx2,
    macz_avx512,
);

bench_variants!(
    emv => EmvInputS; None;
    emv_scalar,
    emv_avx2,
    emv_avx512,
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
    ehlers_pma => EhlersPmaInputS; None;
    ehlers_pma_scalar,
    ehlers_pma_avx2,
    ehlers_pma_avx512,
);

bench_variants!(
    ehlers_kama => EhlersKamaInputS; None;
    ehlers_kama_scalar,
    ehlers_kama_avx2,
    ehlers_kama_avx512,
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
    pma => PmaInputS; None;
    pma_scalar,
    pma_avx2,
    pma_avx512,
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

bench_variants!(
    gaussian => GaussianInputS; None;
    gaussian_scalar,
    gaussian_avx2,
    gaussian_avx512,
);

// Pivot single-series variants
bench_variants!(
    pivot => PivotInputS; None;
    pivot_scalar,
    pivot_avx2,
    pivot_avx512,
);

bench_variants!(
    highpass_2_pole => HighPass2InputS; None;
    highpass_2_pole_scalar,
    highpass_2_pole_avx2,
    highpass_2_pole_avx512,
);

bench_variants!(
    highpass => HighPassInputS; None;
    highpass_scalar,
    highpass_avx2,
    highpass_avx512,
);

bench_variants!(
    hma => HmaInputS; None;
    hma_scalar,
    hma_avx2,
    hma_avx512,
);

bench_variants!(
    hwma => HwmaInputS; None;
    hwma_scalar,
    hwma_avx2,
    hwma_avx512,
);

bench_variants!(
    jma => JmaInputS; None;
    jma_scalar,
    jma_avx2,
    jma_avx512,
);

bench_variants!(
    jsa => JsaInputS; None;
    jsa_scalar,
    jsa_avx2,
    jsa_avx512,
);

bench_variants!(
    kama => KamaInputS; None;
    kama_scalar,
    kama_avx2,
    kama_avx512,
);

bench_variants!(
    linreg => LinRegInputS; None;
    linreg_scalar,
    linreg_avx2,
    linreg_avx512,
);

bench_variants!(
    maaq => MaaqInputS; None;
    maaq_scalar,
    maaq_avx2,
    maaq_avx512,
);

bench_variants!(
    mama => MamaInputS; None;
    mama_scalar,
    mama_avx2,
    mama_avx512,
);

bench_variants!(
    mwdx => MwdxInputS; None;
    mwdx_scalar,
    mwdx_avx2,
    mwdx_avx512,
);

bench_variants!(
    nma => NmaInputS; None;
    nma_scalar,
    nma_avx2,
    nma_avx512,
);

bench_variants!(
    pwma => PwmaInputS; None;
    pwma_scalar,
    pwma_avx2,
    pwma_avx512,
);

bench_variants!(
    reflex => ReflexInputS; None;
    reflex_scalar,
    reflex_avx2,
    reflex_avx512,
);

bench_variants!(
    sinwma => SinWmaInputS; None;
    sinwma_scalar,
    sinwma_avx2,
    sinwma_avx512,
);

bench_variants!(
    sma => SmaInputS; None;
    sma_scalar,
    sma_avx2,
    sma_avx512,
);

bench_variants!(
    smma => SmmaInputS; None;
    smma_scalar,
    smma_avx2,
    smma_avx512,
);

bench_variants!(
    sqwma => SqwmaInputS; None;
    sqwma_scalar,
    sqwma_avx2,
    sqwma_avx512,
);

bench_variants!(
    srwma => SrwmaInputS; None;
    srwma_scalar,
    srwma_avx2,
    srwma_avx512,
);

bench_variants!(
    supersmoother => SuperSmootherInputS; None;
    supersmoother_scalar,
    supersmoother_avx2,
    supersmoother_avx512,
);

bench_variants!(
    supersmoother_3_pole => SuperSmoother3PoleInputS; None;
    supersmoother_3_pole_scalar,
    supersmoother_3_pole_avx2,
    supersmoother_3_pole_avx512,
);

bench_variants!(
    swma => SwmaInputS; None;
    swma_scalar,
    swma_avx2,
    swma_avx512,
);

bench_variants!(
    tema => TemaInputS; None;
    tema_scalar,
    tema_avx2,
    tema_avx512,
);

bench_variants!(
    tilson => TilsonInputS; None;
    tilson_scalar,
    tilson_avx2,
    tilson_avx512,
);

bench_variants!(
    tradjema => TradjemaInputS; None;
    tradjema_scalar,
    tradjema_avx2,
    tradjema_avx512,
);

bench_variants!(
    trendflex => TrendFlexInputS; None;
    trendflex_scalar,
    trendflex_avx2,
    trendflex_avx512,
);

bench_variants!(
    trima => TrimaInputS; None;
    trima_scalar,
    trima_avx2,
    trima_avx512,
);

bench_variants!(
    uma => UmaInputS; None;
    uma_scalar,
    uma_avx2,
    uma_avx512,
);

bench_variants!(
    chandelier_exit => ChandelierExitInputS; None;
    chandelier_exit_scalar,
    chandelier_exit_avx2,
    chandelier_exit_avx512,
);

bench_variants!(
    percentile_nearest_rank => PercentileNearestRankInputS; None;
    percentile_nearest_rank_scalar,
    percentile_nearest_rank_avx2,
    percentile_nearest_rank_avx512,
);

bench_variants!(
    vidya => VidyaInputS; None;
    vidya_scalar,
    vidya_avx2,
    vidya_avx512,
);

bench_variants!(
    vlma => VlmaInputS; None;
    vlma_scalar,
    vlma_avx2,
    vlma_avx512,
);

bench_variants!(
    volume_adjusted_ma => VolumeAdjustedMaInputS; None;
    volume_adjusted_ma_scalar,
    volume_adjusted_ma_avx2,
    volume_adjusted_ma_avx512,
);

bench_variants!(
    vpwma => VpwmaInputS; None;
    vpwma_scalar,
    vpwma_avx2,
    vpwma_avx512,
);

bench_variants!(
    vwma => VwmaInputS; None;
    vwma_scalar,
    vwma_avx2,
    vwma_avx512,
);

bench_variants!(
    wilders => WildersInputS; None;
    wilders_scalar,
    wilders_avx2,
    wilders_avx512,
);

// Stochastic Oscillator (single-series) variants
bench_variants!(
    stoch => StochInputS; None;
    stoch_scalar,
    stoch_avx2,
    stoch_avx512,
);

// Stochastic Oscillator (batch) variants (use default params; 1 row)
bench_variants!(
    stoch_batch => StochInputS; Some(14);
    stoch_batch_scalarbatch,
    stoch_batch_avx2batch,
    stoch_batch_avx512batch,
);

bench_variants!(
    wclprice => WclpriceInputS; None;
    wclprice_scalar,
    wclprice_avx2,
    wclprice_avx512,
);

bench_variants!(
    wma => WmaInputS; None;
    wma_scalar,
    wma_avx2,
    wma_avx512,
);

bench_variants!(
    zlema => ZlemaInputS; None;
    zlema_scalar,
    zlema_avx2,
    zlema_avx512,
);

bench_variants!(
    donchian => DonchianInputS; None;
    donchian_scalar,
    donchian_avx2,
    donchian_avx512,
);

bench_variants!(
    donchian_batch => DonchianInputS; Some(20);
    donchian_batch_scalarbatch,
    donchian_batch_avx2batch,
    donchian_batch_avx512batch,
);

bench_variants!(
    stochf => StochfInputS; None;
    stochf_scalar,
    stochf_avx2,
    stochf_avx512,
);

// Other indicators single variants
bench_variants!(
    avsl => AvslInputS; None;
    avsl_scalar,
    avsl_avx2,
    avsl_avx512,
);

bench_variants!(
    dma => DmaInputS; None;
    dma_scalar,
    dma_avx2,
    dma_avx512,
);

bench_variants!(
    range_filter => RangeFilterInputS; None;
    range_filter_scalar,
    range_filter_avx2,
    range_filter_avx512,
);

bench_variants!(
    ehma => EhmaInputS; None;
    ehma_scalar,
    ehma_avx2,
    ehma_avx512,
);

bench_variants!(
    sama => SamaInputS; None;
    sama_scalar,
    sama_avx2,
    sama_avx512,
);

bench_variants!(
    wto => WtoInputS; None;
    wto_scalar,
    wto_avx2,
    wto_avx512,
);

bench_variants!(
    wavetrend => WavetrendInputS; None;
    wavetrend_scalar,
    wavetrend_avx2,
    wavetrend_avx512,
);

bench_variants!(
    dpo => DpoInputS; None;
    dpo_scalar,
    dpo_avx2,
    dpo_avx512,
);

bench_variants!(
    dpo_batch => DpoInputS; None;
    dpo_batch_scalarbatch,
    dpo_batch_avx2batch,
    dpo_batch_avx512batch,
);

bench_variants!(
    fosc => FoscInputS; None;
    fosc_scalar,
    fosc_avx2,
    fosc_avx512,
);

bench_variants!(
    wavetrend_batch => WavetrendInputS; None;
    wavetrend_batch_scalarbatch,
    wavetrend_batch_avx2batch,
    wavetrend_batch_avx512batch,
);

bench_variants!(
    tsf => TsfInputS; None;
    tsf_scalar,
    tsf_avx2,
    tsf_avx512,
);

bench_variants!(
    vosc_batch => VoscInputS; Some(227);
    vosc_batch_scalarbatch,
    vosc_batch_avx2batch,
    vosc_batch_avx512batch,
);

bench_variants!(
    nama => NamaInputS; None;
    nama_scalar,
    nama_avx2,
    nama_avx512,
);

bench_variants!(
    deviation => DeviationInputS; Some(9);
    deviation_scalar,
    deviation_avx2,
    deviation_avx512,
);

bench_variants!(
    deviation_batch => DeviationInputS; Some(9);
    deviation_batch_scalarbatch,
    deviation_batch_avx2batch,
    deviation_batch_avx512batch,
);

bench_variants!(
    mom => MomInputS; None;
    mom_scalar,
    mom_avx2,
    mom_avx512,
);

bench_variants!(
    supertrend => SupertrendInputS; Some(10);
    supertrend_scalar,
    supertrend_avx2,
    supertrend_avx512,
);

// RSX single-series variants (to compare scalar vs AVX2/AVX512)
bench_variants!(
    rsx => RsxInputS; None;
    rsx_scalar,
    rsx_avx2,
    rsx_avx512,
);

// RSX batch variants (advisory; SIMD stubs delegate to scalar)
bench_variants!(
    rsx_batch => RsxInputS; None;
    rsx_batch_scalarbatch,
    rsx_batch_avx2batch,
    rsx_batch_avx512batch
);

bench_variants!(
    net_myrsi => NetMyrsiInputS; None;
    net_myrsi_scalar,
    net_myrsi_avx2,
    net_myrsi_avx512,
);

bench_variants!(
    net_myrsi_batch => NetMyrsiInputS; Some(37);
    net_myrsi_batch_scalarbatch,
    net_myrsi_batch_avx2batch,
    net_myrsi_batch_avx512batch
);

bench_variants!(
    cci_cycle => CciCycleInputS; None;
    cci_cycle_scalar,
    cci_cycle_avx2,
    cci_cycle_avx512,
);

bench_variants!(
    cci_cycle_batch => CciCycleInputS; Some(227);
    cci_cycle_batch_scalarbatch,
    cci_cycle_batch_avx2batch,
    cci_cycle_batch_avx512batch
);

bench_variants!(
    fvg_trailing_stop => FvgTrailingStopInputS; None;
    fvg_trailing_stop_scalar,
    fvg_trailing_stop_avx2,
    fvg_trailing_stop_avx512,
);

// TODO: FvgTrailingStop needs custom batch wrapper
// bench_variants!(
// 	fvg_trailing_stop_batch => FvgTrailingStopInputS; Some(100);
// 	fvg_trailing_stop_batch_scalarbatch,
// 	fvg_trailing_stop_batch_avx2batch,
// 	fvg_trailing_stop_batch_avx512batch
// );

bench_variants!(
    halftrend => HalfTrendInputS; None;
    halftrend_scalar,
    halftrend_avx2,
    halftrend_avx512,
);

// TODO: HalfTrend needs custom batch wrapper
// bench_variants!(
// 	halftrend_batch => HalfTrendInputS; Some(100);
// 	halftrend_batch_scalarbatch,
// 	halftrend_batch_avx2batch,
// 	halftrend_batch_avx512batch
// );

bench_variants!(
    reverse_rsi => ReverseRsiInputS; None;
    reverse_rsi_scalar,
    reverse_rsi_avx2,
    reverse_rsi_avx512,
);

// Medium AD single-series variants
bench_variants!(
    medium_ad => MediumAdInputS; None;
    medium_ad_scalar,
    medium_ad_avx2,
    medium_ad_avx512,
);

// MSW single-series: compare Scalar vs AVX2 vs AVX512
bench_variants!(
    msw => MswInputS; None;
    msw_scalar,
    msw_avx2,
    msw_avx512,
);

bench_variants!(
    reverse_rsi_batch => ReverseRsiInputS; Some(27);
    reverse_rsi_batch_scalarbatch,
    reverse_rsi_batch_avx2batch,
    reverse_rsi_batch_avx512batch
);

bench_variants!(
    vama => VamaInputS; None;
    vama_scalar,
    vama_avx2,
    vama_avx512,
);

bench_variants!(
    vama_batch => VamaInputS; Some(232);
    vama_batch_scalarbatch,
    vama_batch_avx2batch,
    vama_batch_avx512batch
);

bench_variants!(
    ppo => PpoInputS; None;
    ppo_scalar,
    ppo_avx2,
    ppo_avx512,
);

bench_variants!(
    kurtosis => KurtosisInputS; None;
    kurtosis_scalar,
    kurtosis_avx2,
    kurtosis_avx512,
);

bench_variants!(
    adxr => AdxrInputS; None;
    adxr_scalar,
    adxr_avx2,
    adxr_avx512,
);

bench_variants!(
    cci => CciInputS; Some(14);
    cci_scalar,
    cci_avx2,
    cci_avx512,
);

bench_variants!(
    kaufmanstop => KaufmanstopInputS; None;
    kaufmanstop_scalar,
    kaufmanstop_avx2,
    kaufmanstop_avx512,
);

bench_variants!(
    cksp => CkspInputS; None;
    cksp_scalar,
    cksp_avx2,
    cksp_avx512,
);

bench_variants!(
    mod_god_mode => ModGodModeInputS; None;
    mod_god_mode_scalar,
    mod_god_mode_avx2,
    mod_god_mode_avx512,
);

bench_variants!(
    cksp_batch => CkspInputS; None;
    cksp_batch_scalarbatch,
    cksp_batch_avx2batch,
    cksp_batch_avx512batch
);

bench_variants!(
    prb_batch => PrbInputS; None;
    prb_batch_scalarbatch,
    prb_batch_avx2batch,
    prb_batch_avx512batch
);

// Aroon Oscillator kernel variant benches (Scalar/Avx2/Avx512)
make_kernel_wrappers!(
    aroon_osc,
    my_project::indicators::aroonosc::aroon_osc_with_kernel,
    AroonOscInputS;
    Scalar, Avx2, Avx512
);

bench_variants!(
    aroon_osc => AroonOscInputS; Some(14);
    aroon_osc_scalar,
    aroon_osc_avx2,
    aroon_osc_avx512
);

// Aroon Oscillator batch benches (needs high+low)
#[inline(always)]
fn aroon_osc_batch_scalarbatch(input: &AroonOscInputS) -> anyhow::Result<()> {
    use my_project::indicators::aroonosc::AroonOscBatchBuilder;
    let high = input.get_high();
    let low = input.get_low();
    AroonOscBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .apply_slices(high, low)
        .map(|_| ())
        .map_err(|e| anyhow::anyhow!(e.to_string()))
}

#[inline(always)]
fn aroon_osc_batch_avx2batch(input: &AroonOscInputS) -> anyhow::Result<()> {
    use my_project::indicators::aroonosc::AroonOscBatchBuilder;
    let high = input.get_high();
    let low = input.get_low();
    AroonOscBatchBuilder::new()
        .kernel(Kernel::Avx2Batch)
        .apply_slices(high, low)
        .map(|_| ())
        .map_err(|e| anyhow::anyhow!(e.to_string()))
}

#[inline(always)]
fn aroon_osc_batch_avx512batch(input: &AroonOscInputS) -> anyhow::Result<()> {
    use my_project::indicators::aroonosc::AroonOscBatchBuilder;
    let high = input.get_high();
    let low = input.get_low();
    AroonOscBatchBuilder::new()
        .kernel(Kernel::Avx512Batch)
        .apply_slices(high, low)
        .map(|_| ())
        .map_err(|e| anyhow::anyhow!(e.to_string()))
}

bench_variants!(
    aroon_osc_batch => AroonOscInputS; Some(14);
    aroon_osc_batch_scalarbatch,
    aroon_osc_batch_avx2batch,
    aroon_osc_batch_avx512batch
);

bench_variants!(
    cmo_batch => CmoInputS; Some(14);
    cmo_batch_scalarbatch,
    cmo_batch_avx2batch,
    cmo_batch_avx512batch
);

bench_variants!(
    srsi => SrsiInputS; Some(14);
    srsi_scalar,
    srsi_avx2,
    srsi_avx512,
);

bench_variants!(
    srsi_batch => SrsiInputS; Some(14);
    srsi_batch_scalarbatch,
    srsi_batch_avx2batch,
    srsi_batch_avx512batch
);

bench_variants!(
    pfe_batch => PfeInputS; Some(10);
    pfe_batch_scalarbatch,
    pfe_batch_avx2batch,
    pfe_batch_avx512batch
);

// NVI single-series kernel variants
bench_variants!(
    nvi => NviInputS; None;
    nvi_scalar,
    nvi_avx2,
    nvi_avx512,
);

bench_variants!(
    cvi => CviInputS; None;
    cvi_scalar,
    cvi_avx2,
    cvi_avx512,
);

bench_variants!(
    cvi_batch => CviInputS; Some(20);
    cvi_batch_scalarbatch,
    cvi_batch_avx2batch,
    cvi_batch_avx512batch,
);

// UI single-series variants (no fixed window accounting for throughput)
// Gator Oscillator single-series variants
bench_variants!(
    gatorosc => GatorOscInputS; None;
    gatorosc_scalar,
    gatorosc_avx2,
    gatorosc_avx512,
);

bench_variants!(
    ui => UiInputS; None;
    ui_scalar,
    ui_avx2,
    ui_avx512,
);

// DVDIQQE single-series variants (Scalar vs AVX2/AVX512)
bench_variants!(
    dvdiqqe => DvdiqqeInputS; None;
    dvdiqqe_scalar,
    dvdiqqe_avx2,
    dvdiqqe_avx512,
);

// UI batch variants
make_batch_wrappers!(
    ui_batch,
    my_project::indicators::ui::UiBatchBuilder,
    UiInputS;
    ScalarBatch,
    Avx2Batch,
    Avx512Batch
);
bench_variants!(
    ui_batch => UiInputS; Some(27);
    ui_batch_scalarbatch,
    ui_batch_avx2batch,
    ui_batch_avx512batch
);

// DI batch variants (OHLC slices required)
#[inline(always)]
fn di_batch_scalarbatch(input: &DiInputS) -> anyhow::Result<()> {
    let (high, low, close) = match &input.data {
        my_project::indicators::di::DiData::Candles { candles } => (
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        ),
        my_project::indicators::di::DiData::Slices { high, low, close } => (*high, *low, *close),
    };
    my_project::indicators::di::DiBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .apply_slices(high, low, close)?;
    Ok(())
}
#[inline(always)]
fn di_batch_avx2batch(input: &DiInputS) -> anyhow::Result<()> {
    let (high, low, close) = match &input.data {
        my_project::indicators::di::DiData::Candles { candles } => (
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        ),
        my_project::indicators::di::DiData::Slices { high, low, close } => (*high, *low, *close),
    };
    my_project::indicators::di::DiBatchBuilder::new()
        .kernel(Kernel::Avx2Batch)
        .apply_slices(high, low, close)?;
    Ok(())
}
#[inline(always)]
fn di_batch_avx512batch(input: &DiInputS) -> anyhow::Result<()> {
    let (high, low, close) = match &input.data {
        my_project::indicators::di::DiData::Candles { candles } => (
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        ),
        my_project::indicators::di::DiData::Slices { high, low, close } => (*high, *low, *close),
    };
    my_project::indicators::di::DiBatchBuilder::new()
        .kernel(Kernel::Avx512Batch)
        .apply_slices(high, low, close)?;
    Ok(())
}
bench_variants!(
    di_batch => DiInputS; Some(27);
    di_batch_scalarbatch,
    di_batch_avx2batch,
    di_batch_avx512batch
);

// MSW batch variants
make_batch_wrappers!(
    msw_batch,
    my_project::indicators::msw::MswBatchBuilder,
    MswInputS;
    ScalarBatch,
    Avx2Batch,
    Avx512Batch
);
bench_variants!(
    msw_batch => MswInputS; Some(27);
    msw_batch_scalarbatch,
    msw_batch_avx2batch,
    msw_batch_avx512batch
);

// Medium AD batch variants
make_batch_wrappers!(
    medium_ad_batch,
    my_project::indicators::medium_ad::MediumAdBatchBuilder,
    MediumAdInputS;
    ScalarBatch,
    Avx2Batch,
    Avx512Batch
);
bench_variants!(
    medium_ad_batch => MediumAdInputS; Some(27);
    medium_ad_batch_scalarbatch,
    medium_ad_batch_avx2batch,
    medium_ad_batch_avx512batch
);

bench_variants!(
    ultosc => UltOscInputS; None;
    ultosc_scalar,
    ultosc_avx2,
    ultosc_avx512,
);

bench_variants!(
    qstick => QstickInputS; None;
    qstick_scalar,
    qstick_avx2,
    qstick_avx512,
);

bench_variants!(
    ultosc_batch => UltOscInputS; Some(28);
    ultosc_batch_scalarbatch,
    ultosc_batch_avx2batch,
    ultosc_batch_avx512batch,
);

bench_variants!(
    qstick_batch => QstickInputS; Some(5);
    qstick_batch_scalarbatch,
    qstick_batch_avx2batch,
    qstick_batch_avx512batch,
);

// MFI batch benchmarks
bench_variants!(
    mfi_batch => MfiInputS; Some(14);
    mfi_batch_scalarbatch,
    mfi_batch_avx2batch,
    mfi_batch_avx512batch,
);

// DX batch benchmarks (period sweep 10..=30 step 5; window representative: 14)
bench_variants!(
    dx_batch => DxInputS; Some(14);
    dx_batch_scalarbatch,
);

criterion_main!(
    benches_scalar,
    benches_adxr,
    benches_bandpass,
    benches_cci,
    benches_correl_hl,
    benches_sar,
    benches_natr,
    benches_efi,
    benches_marketfi,
    benches_natr_batch,
    benches_sar_batch,
    benches_stoch,
    benches_stoch_batch,
    benches_fisher_batch,
    // Range Filter (single + batch)
    benches_range_filter,
    benches_range_filter_batch,
    benches_acosc,
    benches_cfo,
    benches_cfo_batch,
    benches_correlation_cycle,
    benches_correlation_cycle_batch,
    benches_kaufmanstop,
    benches_cksp,
    benches_cksp_batch,
    benches_aso,
    benches_alphatrend,
    benches_bollinger_bands,
    benches_cg,
    benches_ad,
    benches_cmo,
    benches_cmo_batch,
    benches_cvi,
    benches_cvi_batch,
    benches_linearreg_angle,
    benches_nvi,
    benches_pvi,
    benches_dti,
    benches_pvi_batch,
    benches_aso_batch,
    benches_mom,
    benches_mom_batch,
    benches_rsx,
    benches_rsx_batch,
    benches_ao,
    benches_ao_batch,
    benches_atr,
    benches_atr_batch,
    benches_er,
    benches_obv,
    benches_trix,
    benches_trix_batch,
    benches_apo,
    benches_apo_batch,
    benches_rocr,
    benches_dm,
    benches_dm_batch,
    benches_linearreg_slope,
    benches_ott,
    benches_ultosc,
    benches_qstick,
    benches_mfi_batch,
    benches_dx_batch,
    benches_adx,
    benches_adx_batch,
    benches_ultosc_batch,
    benches_qstick_batch,
    benches_alma,
    benches_adosc,
    benches_nadaraya_watson_envelope,
    benches_alma_batch,
    benches_mod_god_mode,
    benches_adosc_batch,
    benches_macd,
    benches_macd_batch,
    benches_buff_averages,
    benches_buff_averages_batch,
    benches_bandpass_batch,
    benches_decycler_batch,
    benches_zscore,
    benches_mab,
    benches_eri,
    benches_zscore_batch,
    benches_var,
    benches_var_batch,
    benches_deviation,
    benches_deviation_batch,
    benches_supertrend,
    benches_bollinger_bands_batch,
    benches_linearreg_slope_batch,
    benches_stddev_batch,
    benches_macz,
    benches_emv,
    benches_macz_batch,
    benches_cwma,
    benches_cwma_batch,
    benches_cora_wave_batch,
    benches_dema,
    benches_dema_batch,
    benches_edcf,
    benches_edcf_batch,
    benches_ehlers_ecema,
    benches_ehlers_ecema_batch,
    benches_ehlers_itrend,
    benches_ehlers_itrend_batch,
    benches_ehlers_pma,
    benches_pma,
    benches_ehlers_pma_batch,
    benches_ehlers_kama,
    benches_ehlers_kama_batch,
    benches_ema,
    benches_ema_batch,
    benches_epma,
    benches_epma_batch,
    benches_pma_batch,
    benches_frama,
    benches_frama_batch,
    benches_fwma,
    benches_fwma_batch,
    benches_gaussian,
    benches_gaussian_batch,
    benches_pivot,
    benches_pivot_batch,
    benches_highpass_2_pole,
    benches_highpass_2_pole_batch,
    benches_highpass,
    benches_highpass_batch,
    benches_hma,
    benches_hma_batch,
    benches_hwma,
    benches_hwma_batch,
    benches_jma,
    benches_jma_batch,
    benches_jsa,
    benches_jsa_batch,
    benches_kama,
    benches_kama_batch,
    benches_linreg,
    benches_linreg_batch,
    benches_linearreg_intercept,
    benches_linearreg_intercept_batch,
    benches_maaq,
    benches_maaq_batch,
    benches_mama,
    benches_mama_batch,
    benches_mwdx,
    benches_msw,
    benches_msw_batch,
    benches_mwdx_batch,
    benches_nma,
    benches_nma_batch,
    benches_pwma,
    benches_pwma_batch,
    benches_reflex,
    benches_reflex_batch,
    benches_sinwma,
    benches_sinwma_batch,
    benches_sma,
    benches_sma_batch,
    benches_smma,
    benches_smma_batch,
    benches_sqwma,
    benches_sqwma_batch,
    benches_srwma,
    benches_srwma_batch,
    benches_supersmoother,
    benches_supersmoother_batch,
    benches_supersmoother_3_pole,
    benches_supersmoother_3_pole_batch,
    benches_swma,
    benches_swma_batch,
    benches_tema,
    benches_tema_batch,
    benches_tilson,
    benches_tilson_batch,
    benches_tradjema,
    benches_tradjema_batch,
    benches_trendflex,
    benches_trendflex_batch,
    benches_trima,
    benches_trima_batch,
    benches_uma,
    benches_uma_batch,
    benches_chandelier_exit,
    benches_percentile_nearest_rank,
    benches_vidya,
    benches_vidya_batch,
    benches_vlma,
    benches_vlma_batch,
    benches_volume_adjusted_ma,
    benches_volume_adjusted_ma_batch,
    // DMA (single + batch)
    benches_dma,
    benches_dma_batch,
    benches_vama,
    benches_vama_batch,
    benches_sama,
    benches_sama_batch,
    benches_er_batch,
    benches_vpwma,
    benches_vpwma_batch,
    benches_prb_batch,
    benches_squeeze_momentum_batch,
    benches_srsi,
    benches_srsi_batch,
    benches_pfe_batch,
    benches_kst_batch,
    benches_willr_batch,
    benches_vwma,
    benches_wavetrend,
    benches_nama,
    benches_wto,
    benches_wavetrend_batch,
    benches_dpo,
    benches_dpo_batch,
    benches_fosc,
    benches_wto_batch,
    benches_nama_batch,
    benches_di_batch,
    benches_vosc_batch,
    benches_wilders,
    benches_wclprice,
    benches_wilders_batch,
    benches_ott_batch,
    benches_wma,
    benches_wma_batch,
    benches_zlema,
    benches_zlema_batch,
    benches_stochf,
    benches_vi_batch,
    benches_tsf,
    benches_chandelier_exit,
    benches_chandelier_exit_batch,
    benches_otto_batch,
    benches_percentile_nearest_rank,
    benches_percentile_nearest_rank_batch,
    benches_ppo,
    benches_donchian,
    benches_donchian_batch,
    benches_kurtosis
    ,benches_net_myrsi
    ,benches_net_myrsi_batch
    ,benches_kvo_batch
    ,benches_bollinger_bands_width
    ,benches_bollinger_bands_width_batch
    ,benches_aroon_osc
    ,benches_aroon_osc_batch
    ,benches_ehma
    ,benches_ehma_batch
);
