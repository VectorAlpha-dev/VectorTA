#![cfg(feature = "cuda")]

use cust::memory::{CopyDestination, DeviceBuffer};
use crate::cuda::moving_averages::{
    CudaAlma, CudaCwma, CudaDema, CudaEma, CudaEpma, CudaFwma, CudaHighpass, CudaJsa, CudaMaaq,
    CudaMwdx, CudaNma, CudaPwma, CudaSinwma, CudaSma, CudaSmma, CudaSrwma, CudaSqwma, CudaSuperSmoother,
    CudaSupersmoother3Pole, CudaSwma, CudaTema, CudaTrima, CudaWilders, CudaWma, CudaZlema,
    CudaEdcf, CudaEhlersITrend, CudaHma, CudaJma, CudaCoraWave, CudaFrama, CudaVpwma, CudaVwma,
};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct AlmaParamKey {
    offset_bits: u64,
    sigma_bits: u64,
}

struct AlmaDeviceConsts {
    d_weights: DeviceBuffer<f32>,
    d_inv_norms: DeviceBuffer<f32>,
    max_period: usize,
}

struct CwmaDeviceConsts {
    d_weights: DeviceBuffer<f32>,
    d_ones: DeviceBuffer<f32>,
    max_period: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct CoraWaveParamKey {
    r_multi_bits: u64,
}

struct CoraWaveDeviceConsts {
    d_weights: DeviceBuffer<f32>,
    d_inv_norms: DeviceBuffer<f32>,
    max_period: usize,
}

struct FwmaDeviceConsts {
    d_weights: DeviceBuffer<f32>,
    max_period: usize,
}

struct PwmaDeviceConsts {
    d_weights: DeviceBuffer<f32>,
    max_period: usize,
}

struct SrwmaDeviceConsts {
    d_weights: DeviceBuffer<f32>,
    d_inv_norms: DeviceBuffer<f32>,
    max_wlen: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct VpwmaParamKey {
    power_bits: u64,
}

struct VpwmaDeviceConsts {
    d_weights: DeviceBuffer<f32>,
    d_inv_norms: DeviceBuffer<f32>,
    stride: usize,
}

pub fn supports_vram_kernel_ma(ma_type: &str) -> bool {
    match ma_type.trim().to_ascii_lowercase().as_str() {
        "sma" | "ema" | "wma" | "alma" | "dema" | "tema" | "jsa" | "smma" | "sqwma" | "highpass"
        | "swma" | "trima" | "sinwma" | "epma" | "wilders" | "maaq" | "mwdx" | "cwma" | "fwma"
        | "pwma" | "srwma" | "supersmoother" | "supersmoother_3_pole" | "zlema"
        | "nma" | "hma" | "jma" | "edcf" | "ehlers_itrend" | "cora_wave"
        | "vwma" | "vpwma" | "frama" => true,
        _ => false,
    }
}

pub struct VramMaInputs<'a> {
    pub prices: &'a DeviceBuffer<f32>,
    pub close: &'a DeviceBuffer<f32>,
    pub high: Option<&'a DeviceBuffer<f32>>,
    pub low: Option<&'a DeviceBuffer<f32>>,
    pub volume: Option<&'a DeviceBuffer<f32>>,
}

pub struct VramMaComputer {
    device_id: u32,

    sma: Option<CudaSma>,
    sma_prefix_f64: Option<DeviceBuffer<f64>>,

    vwma: Option<CudaVwma>,
    vwma_pv_prefix_f64: Option<DeviceBuffer<f64>>,
    vwma_vol_prefix_f64: Option<DeviceBuffer<f64>>,

    ema: Option<CudaEma>,
    wma: Option<CudaWma>,
    alma: Option<CudaAlma>,
    dema: Option<CudaDema>,
    tema: Option<CudaTema>,

    jsa: Option<CudaJsa>,
    smma: Option<CudaSmma>,
    sqwma: Option<CudaSqwma>,
    highpass: Option<CudaHighpass>,
    swma: Option<CudaSwma>,
    trima: Option<CudaTrima>,
    sinwma: Option<CudaSinwma>,
    epma: Option<CudaEpma>,
    wilders: Option<CudaWilders>,
    maaq: Option<CudaMaaq>,
    mwdx: Option<CudaMwdx>,
    cwma: Option<CudaCwma>,
    fwma: Option<CudaFwma>,
    pwma: Option<CudaPwma>,
    srwma: Option<CudaSrwma>,
    supersmoother: Option<CudaSuperSmoother>,
    supersmoother_3_pole: Option<CudaSupersmoother3Pole>,
    zlema: Option<CudaZlema>,

    cora_wave: Option<CudaCoraWave>,
    cora_wave_tmp: Option<DeviceBuffer<f32>>,

    nma: Option<CudaNma>,
    nma_abs_diffs: Option<DeviceBuffer<f32>>,

    hma: Option<CudaHma>,
    hma_ring: Option<DeviceBuffer<f32>>,

    jma: Option<CudaJma>,
    edcf: Option<CudaEdcf>,
    edcf_dist: Option<DeviceBuffer<f32>>,
    ehlers_itrend: Option<CudaEhlersITrend>,

    vpwma: Option<CudaVpwma>,
    frama: Option<CudaFrama>,

    host_i32: Vec<i32>,
    host_i32_b: Vec<i32>,
    host_f32: Vec<f32>,
    host_f32_b: Vec<f32>,
    host_f32_c: Vec<f32>,

    d_i32: Option<DeviceBuffer<i32>>,
    d_i32_b: Option<DeviceBuffer<i32>>,
    d_f32: Option<DeviceBuffer<f32>>,
    d_f32_b: Option<DeviceBuffer<f32>>,
    d_f32_c: Option<DeviceBuffer<f32>>,

    alma_cache: HashMap<AlmaParamKey, HashMap<Box<[i32]>, AlmaDeviceConsts>>,
    cwma_cache: HashMap<Box<[i32]>, CwmaDeviceConsts>,
    cora_wave_cache: HashMap<CoraWaveParamKey, HashMap<Box<[i32]>, CoraWaveDeviceConsts>>,
    fwma_cache: HashMap<Box<[i32]>, FwmaDeviceConsts>,
    pwma_cache: HashMap<Box<[i32]>, PwmaDeviceConsts>,
    srwma_cache: HashMap<Box<[i32]>, SrwmaDeviceConsts>,
    vpwma_cache: HashMap<VpwmaParamKey, HashMap<Box<[i32]>, VpwmaDeviceConsts>>,
}

impl VramMaComputer {
    pub fn new(device_id: u32) -> Self {
        Self {
            device_id,

            sma: None,
            sma_prefix_f64: None,

            vwma: None,
            vwma_pv_prefix_f64: None,
            vwma_vol_prefix_f64: None,

            ema: None,
            wma: None,
            alma: None,
            dema: None,
            tema: None,

            jsa: None,
            smma: None,
            sqwma: None,
            highpass: None,
            swma: None,
            trima: None,
            sinwma: None,
            epma: None,
            wilders: None,
            maaq: None,
            mwdx: None,
            cwma: None,
            fwma: None,
            pwma: None,
            srwma: None,
            supersmoother: None,
            supersmoother_3_pole: None,
            zlema: None,

            cora_wave: None,
            cora_wave_tmp: None,

            nma: None,
            nma_abs_diffs: None,

            hma: None,
            hma_ring: None,

            jma: None,
            edcf: None,
            edcf_dist: None,
            ehlers_itrend: None,

            vpwma: None,
            frama: None,

            host_i32: Vec::new(),
            host_i32_b: Vec::new(),
            host_f32: Vec::new(),
            host_f32_b: Vec::new(),
            host_f32_c: Vec::new(),

            d_i32: None,
            d_i32_b: None,
            d_f32: None,
            d_f32_b: None,
            d_f32_c: None,

            alma_cache: HashMap::new(),
            cwma_cache: HashMap::new(),
            cora_wave_cache: HashMap::new(),
            fwma_cache: HashMap::new(),
            pwma_cache: HashMap::new(),
            srwma_cache: HashMap::new(),
            vpwma_cache: HashMap::new(),
        }
    }

    pub fn clear_cached_constants(&mut self) {
        self.alma_cache.clear();
        self.cwma_cache.clear();
        self.cora_wave_cache.clear();
        self.fwma_cache.clear();
        self.pwma_cache.clear();
        self.srwma_cache.clear();
        self.vpwma_cache.clear();
    }

    fn ensure_sma(&mut self) -> Result<(), String> {
        if self.sma.is_none() {
            self.sma = Some(CudaSma::new(self.device_id as usize).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_vwma(&mut self) -> Result<(), String> {
        if self.vwma.is_none() {
            self.vwma = Some(CudaVwma::new(self.device_id as usize).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_ema(&mut self) -> Result<(), String> {
        if self.ema.is_none() {
            self.ema = Some(CudaEma::new(self.device_id as usize).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_wma(&mut self) -> Result<(), String> {
        if self.wma.is_none() {
            self.wma = Some(CudaWma::new(self.device_id as usize).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_alma(&mut self) -> Result<(), String> {
        if self.alma.is_none() {
            self.alma = Some(CudaAlma::new(self.device_id as usize).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_dema(&mut self) -> Result<(), String> {
        if self.dema.is_none() {
            self.dema = Some(CudaDema::new(self.device_id as usize).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_tema(&mut self) -> Result<(), String> {
        if self.tema.is_none() {
            self.tema = Some(CudaTema::new(self.device_id as usize).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_jsa(&mut self) -> Result<(), String> {
        if self.jsa.is_none() {
            self.jsa = Some(CudaJsa::new(self.device_id as usize).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_smma(&mut self) -> Result<(), String> {
        if self.smma.is_none() {
            self.smma =
                Some(CudaSmma::new(self.device_id as usize).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_sqwma(&mut self) -> Result<(), String> {
        if self.sqwma.is_none() {
            self.sqwma =
                Some(CudaSqwma::new(self.device_id as usize).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_highpass(&mut self) -> Result<(), String> {
        if self.highpass.is_none() {
            self.highpass =
                Some(CudaHighpass::new(self.device_id as usize).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_swma(&mut self) -> Result<(), String> {
        if self.swma.is_none() {
            self.swma =
                Some(CudaSwma::new(self.device_id as usize).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_trima(&mut self) -> Result<(), String> {
        if self.trima.is_none() {
            self.trima =
                Some(CudaTrima::new(self.device_id as usize).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_sinwma(&mut self) -> Result<(), String> {
        if self.sinwma.is_none() {
            self.sinwma =
                Some(CudaSinwma::new(self.device_id as usize).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_epma(&mut self) -> Result<(), String> {
        if self.epma.is_none() {
            self.epma =
                Some(CudaEpma::new(self.device_id as usize).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_wilders(&mut self) -> Result<(), String> {
        if self.wilders.is_none() {
            self.wilders =
                Some(CudaWilders::new(self.device_id as usize).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_maaq(&mut self) -> Result<(), String> {
        if self.maaq.is_none() {
            self.maaq =
                Some(CudaMaaq::new(self.device_id as usize).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_mwdx(&mut self) -> Result<(), String> {
        if self.mwdx.is_none() {
            self.mwdx =
                Some(CudaMwdx::new(self.device_id as usize).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_cwma(&mut self) -> Result<(), String> {
        if self.cwma.is_none() {
            self.cwma =
                Some(CudaCwma::new(self.device_id as usize).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_fwma(&mut self) -> Result<(), String> {
        if self.fwma.is_none() {
            self.fwma =
                Some(CudaFwma::new(self.device_id as usize).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_pwma(&mut self) -> Result<(), String> {
        if self.pwma.is_none() {
            self.pwma =
                Some(CudaPwma::new(self.device_id as usize).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_srwma(&mut self) -> Result<(), String> {
        if self.srwma.is_none() {
            self.srwma =
                Some(CudaSrwma::new(self.device_id as usize).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_supersmoother_3_pole(&mut self) -> Result<(), String> {
        if self.supersmoother_3_pole.is_none() {
            self.supersmoother_3_pole = Some(
                CudaSupersmoother3Pole::new(self.device_id as usize).map_err(|e| e.to_string())?,
            );
        }
        Ok(())
    }

    fn ensure_supersmoother(&mut self) -> Result<(), String> {
        if self.supersmoother.is_none() {
            self.supersmoother = Some(
                CudaSuperSmoother::new(self.device_id as usize).map_err(|e| e.to_string())?,
            );
        }
        Ok(())
    }

    fn ensure_zlema(&mut self) -> Result<(), String> {
        if self.zlema.is_none() {
            self.zlema =
                Some(CudaZlema::new(self.device_id as usize).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_cora_wave(&mut self) -> Result<(), String> {
        if self.cora_wave.is_none() {
            self.cora_wave = Some(
                CudaCoraWave::new(self.device_id as usize).map_err(|e| e.to_string())?,
            );
        }
        Ok(())
    }

    fn ensure_nma(&mut self) -> Result<(), String> {
        if self.nma.is_none() {
            self.nma = Some(CudaNma::new(self.device_id as usize).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_hma(&mut self) -> Result<(), String> {
        if self.hma.is_none() {
            self.hma = Some(CudaHma::new(self.device_id as usize).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_jma(&mut self) -> Result<(), String> {
        if self.jma.is_none() {
            self.jma = Some(CudaJma::new(self.device_id as usize).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_edcf(&mut self) -> Result<(), String> {
        if self.edcf.is_none() {
            self.edcf = Some(CudaEdcf::new(self.device_id as usize).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_ehlers_itrend(&mut self) -> Result<(), String> {
        if self.ehlers_itrend.is_none() {
            self.ehlers_itrend = Some(
                CudaEhlersITrend::new(self.device_id as usize).map_err(|e| e.to_string())?,
            );
        }
        Ok(())
    }

    fn ensure_vpwma(&mut self) -> Result<(), String> {
        if self.vpwma.is_none() {
            self.vpwma = Some(CudaVpwma::new(self.device_id as usize).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_frama(&mut self) -> Result<(), String> {
        if self.frama.is_none() {
            self.frama = Some(CudaFrama::new(self.device_id as usize).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_len_i32(buf: &mut Option<DeviceBuffer<i32>>, len: usize) -> Result<(), String> {
        if buf.as_ref().map(|b| b.len()) != Some(len) {
            *buf = Some(unsafe { DeviceBuffer::<i32>::uninitialized(len) }.map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_len_f32(buf: &mut Option<DeviceBuffer<f32>>, len: usize) -> Result<(), String> {
        if buf.as_ref().map(|b| b.len()) != Some(len) {
            *buf = Some(unsafe { DeviceBuffer::<f32>::uninitialized(len) }.map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_len_f64(buf: &mut Option<DeviceBuffer<f64>>, len: usize) -> Result<(), String> {
        if buf.as_ref().map(|b| b.len()) != Some(len) {
            *buf = Some(unsafe { DeviceBuffer::<f64>::uninitialized(len) }.map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn get_param_f64(params: Option<&HashMap<String, f64>>, key: &str) -> Option<f64> {
        params.and_then(|m| m.get(key).copied()).filter(|v| v.is_finite())
    }

    pub fn ensure_sma_prefix_f64(
        &mut self,
        d_prices: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
    ) -> Result<(), String> {
        self.ensure_sma()?;
        if self.sma_prefix_f64.as_ref().map(|b| b.len()) != Some(series_len + 1) {
            self.sma_prefix_f64 = Some(
                unsafe { DeviceBuffer::<f64>::uninitialized(series_len + 1) }.map_err(|e| e.to_string())?,
            );
        }
        let sma = self.sma.as_ref().unwrap();
        sma.sma_prefix_f64_device_into(
            d_prices,
            series_len,
            first_valid,
            self.sma_prefix_f64.as_mut().unwrap(),
        )
        .map_err(|e| e.to_string())?;
        sma.synchronize().map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn ensure_vwma_prefix_pv_vol_f64(
        &mut self,
        d_prices: &DeviceBuffer<f32>,
        d_volumes: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
    ) -> Result<(), String> {
        self.ensure_vwma()?;
        Self::ensure_len_f64(&mut self.vwma_pv_prefix_f64, series_len)?;
        Self::ensure_len_f64(&mut self.vwma_vol_prefix_f64, series_len)?;

        let vwma = self.vwma.as_ref().unwrap();
        vwma.vwma_prefix_pv_vol_f64_device_into(
            d_prices,
            d_volumes,
            series_len,
            first_valid,
            self.vwma_pv_prefix_f64.as_mut().unwrap(),
            self.vwma_vol_prefix_f64.as_mut().unwrap(),
        )
        .map_err(|e| e.to_string())?;
        vwma.synchronize().map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn compute_period_ma_into(
        &mut self,
        ma_type: &str,
        params: Option<&HashMap<String, f64>>,
        inputs: &VramMaInputs,
        series_len: usize,
        first_valid: usize,
        periods: &[i32],
        d_periods: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), String> {
        let ma = ma_type.trim().to_ascii_lowercase();
        let n_combos = periods.len();
        if n_combos == 0 {
            return Err("empty period list".to_string());
        }
        if d_periods.len() < n_combos {
            return Err("device periods buffer too small".to_string());
        }
        if d_out.len() < n_combos.saturating_mul(series_len) {
            return Err("device output buffer too small".to_string());
        }
        if inputs.prices.len() != series_len {
            return Err("device prices buffer length mismatch".to_string());
        }
        if inputs.close.len() != series_len {
            return Err("device close buffer length mismatch".to_string());
        }
        let d_prices = inputs.prices;

        match ma.as_str() {
            "sma" => {
                self.ensure_sma()?;
                let prefix = self
                    .sma_prefix_f64
                    .as_ref()
                    .ok_or_else(|| "sma prefix missing (call ensure_sma_prefix_f64 first)".to_string())?;
                let sma = self.sma.as_ref().unwrap();
                sma.sma_batch_from_prefix_f64_device_into(
                    prefix,
                    d_periods,
                    series_len,
                    n_combos,
                    first_valid,
                    d_out,
                )
                .map_err(|e| e.to_string())?;
                sma.synchronize().map_err(|e| e.to_string())?;
                Ok(())
            }

            "ema" => {
                self.ensure_ema()?;
                self.host_f32.clear();
                self.host_f32
                    .extend(periods.iter().map(|&p| 2.0f32 / (p as f32 + 1.0f32)));
                Self::ensure_len_f32(&mut self.d_f32, n_combos)?;
                self.d_f32
                    .as_mut()
                    .unwrap()
                    .copy_from(&self.host_f32)
                    .map_err(|e| e.to_string())?;

                let ema = self.ema.as_ref().unwrap();
                ema.ema_batch_device(
                    d_prices,
                    d_periods,
                    self.d_f32.as_ref().unwrap(),
                    series_len,
                    first_valid,
                    n_combos,
                    d_out,
                )
                .map_err(|e| e.to_string())?;
                ema.synchronize().map_err(|e| e.to_string())?;
                Ok(())
            }

            "wma" => {
                self.ensure_wma()?;
                let max_period = periods.iter().copied().map(|p| p as usize).max().unwrap_or(0);
                let wma = self.wma.as_ref().unwrap();
                wma.wma_batch_device(
                    d_prices,
                    d_periods,
                    series_len as i32,
                    n_combos as i32,
                    first_valid as i32,
                    max_period as i32,
                    d_out,
                )
                .map_err(|e| e.to_string())?;
                wma.synchronize().map_err(|e| e.to_string())?;
                Ok(())
            }

            "alma" => {
                self.ensure_alma()?;
                let offset = Self::get_param_f64(params, "offset").unwrap_or(0.85);
                let sigma = Self::get_param_f64(params, "sigma").unwrap_or(6.0);
                let key = AlmaParamKey {
                    offset_bits: offset.to_bits(),
                    sigma_bits: sigma.to_bits(),
                };
                if let Some(cached) = self
                    .alma_cache
                    .get(&key)
                    .and_then(|m| m.get(periods))
                {
                    let alma = self.alma.as_ref().unwrap();
                    alma.alma_batch_device(
                        d_prices,
                        &cached.d_weights,
                        d_periods,
                        &cached.d_inv_norms,
                        cached.max_period as i32,
                        series_len as i32,
                        n_combos as i32,
                        first_valid as i32,
                        d_out,
                    )
                    .map_err(|e| e.to_string())?;
                    alma.synchronize().map_err(|e| e.to_string())?;
                    return Ok(());
                }

                let max_period = periods.iter().copied().map(|p| p as usize).max().unwrap_or(0);
                if max_period == 0 {
                    return Err("alma max_period is 0".to_string());
                }

                self.host_f32.clear();
                self.host_f32.resize(n_combos * max_period, 0.0f32);
                self.host_f32_b.clear();
                self.host_f32_b.resize(n_combos, 0.0f32);
                for (combo_idx, &p_i32) in periods.iter().enumerate() {
                    let period = p_i32 as usize;
                    let (mut w, inv_norm) = compute_alma_weights_f32(period, offset, sigma);
                    if inv_norm != 0.0 {
                        for wi in w.iter_mut() {
                            *wi *= inv_norm;
                        }
                    }
                    self.host_f32_b[combo_idx] = 1.0f32;
                    let base = combo_idx * max_period;
                    self.host_f32[base..base + period].copy_from_slice(&w);
                }

                let d_weights = DeviceBuffer::from_slice(&self.host_f32).map_err(|e| e.to_string())?;
                let d_inv_norms = DeviceBuffer::from_slice(&self.host_f32_b).map_err(|e| e.to_string())?;

                let alma = self.alma.as_ref().unwrap();
                alma.alma_batch_device(
                    d_prices,
                    &d_weights,
                    d_periods,
                    &d_inv_norms,
                    max_period as i32,
                    series_len as i32,
                    n_combos as i32,
                    first_valid as i32,
                    d_out,
                )
                .map_err(|e| e.to_string())?;
                alma.synchronize().map_err(|e| e.to_string())?;

                self.alma_cache
                    .entry(key)
                    .or_insert_with(HashMap::new)
                    .insert(
                        periods.to_vec().into_boxed_slice(),
                        AlmaDeviceConsts {
                            d_weights,
                            d_inv_norms,
                            max_period,
                        },
                    );
                Ok(())
            }

            "dema" => {
                self.ensure_dema()?;
                let dema = self.dema.as_ref().unwrap();
                dema.dema_batch_device(
                    d_prices,
                    d_periods,
                    series_len as i32,
                    first_valid as i32,
                    n_combos as i32,
                    d_out,
                )
                .map_err(|e| e.to_string())?;
                dema.synchronize().map_err(|e| e.to_string())?;
                Ok(())
            }

            "tema" => {
                self.ensure_tema()?;
                let tema = self.tema.as_ref().unwrap();
                tema.tema_batch_device(d_prices, d_periods, series_len, n_combos, first_valid, d_out)
                    .map_err(|e| e.to_string())?;
                tema.synchronize().map_err(|e| e.to_string())?;
                Ok(())
            }

            "jsa" => {
                self.ensure_jsa()?;
                self.host_i32.clear();
                self.host_i32
                    .extend(periods.iter().map(|&p| first_valid as i32 + p - 1));
                Self::ensure_len_i32(&mut self.d_i32, n_combos)?;
                self.d_i32
                    .as_mut()
                    .unwrap()
                    .copy_from(&self.host_i32)
                    .map_err(|e| e.to_string())?;

                let jsa = self.jsa.as_ref().unwrap();
                jsa.jsa_batch_device(
                    d_prices,
                    d_periods,
                    self.d_i32.as_ref().unwrap(),
                    series_len,
                    first_valid,
                    n_combos,
                    d_out,
                )
                .map_err(|e| e.to_string())?;
                jsa.synchronize().map_err(|e| e.to_string())?;
                Ok(())
            }

            "smma" => {
                self.ensure_smma()?;
                self.host_i32.clear();
                self.host_i32
                    .extend(periods.iter().map(|&p| first_valid as i32 + p - 1));
                Self::ensure_len_i32(&mut self.d_i32, n_combos)?;
                self.d_i32
                    .as_mut()
                    .unwrap()
                    .copy_from(&self.host_i32)
                    .map_err(|e| e.to_string())?;

                let smma = self.smma.as_ref().unwrap();
                smma.smma_batch_device(
                    d_prices,
                    d_periods,
                    self.d_i32.as_ref().unwrap(),
                    first_valid,
                    series_len,
                    n_combos,
                    d_out,
                )
                .map_err(|e| e.to_string())?;
                smma.synchronize().map_err(|e| e.to_string())?;
                Ok(())
            }

            "sqwma" => {
                self.ensure_sqwma()?;
                let max_period = periods.iter().copied().map(|p| p as usize).max().unwrap_or(0);
                let sqwma = self.sqwma.as_ref().unwrap();
                sqwma.sqwma_batch_device(d_prices, d_periods, series_len, n_combos, first_valid, max_period, d_out)
                    .map_err(|e| e.to_string())?;
                sqwma.synchronize().map_err(|e| e.to_string())?;
                Ok(())
            }

            "highpass" => {
                self.ensure_highpass()?;
                let highpass = self.highpass.as_ref().unwrap();
                highpass
                    .highpass_batch_device(
                        d_prices,
                        first_valid as i32,
                        d_periods,
                        series_len as i32,
                        n_combos as i32,
                        d_out,
                    )
                    .map_err(|e| e.to_string())?;
                highpass.synchronize().map_err(|e| e.to_string())?;
                Ok(())
            }

            "swma" => {
                self.ensure_swma()?;
                let max_period = periods.iter().copied().map(|p| p as usize).max().unwrap_or(0);
                self.host_i32.clear();
                self.host_i32
                    .extend(periods.iter().map(|&p| first_valid as i32 + p - 1));
                Self::ensure_len_i32(&mut self.d_i32, n_combos)?;
                self.d_i32
                    .as_mut()
                    .unwrap()
                    .copy_from(&self.host_i32)
                    .map_err(|e| e.to_string())?;

                let swma = self.swma.as_ref().unwrap();
                swma.swma_batch_device(
                    d_prices,
                    d_periods,
                    self.d_i32.as_ref().unwrap(),
                    series_len,
                    n_combos,
                    max_period,
                    d_out,
                )
                .map_err(|e| e.to_string())?;
                swma.synchronize().map_err(|e| e.to_string())?;
                Ok(())
            }

            "trima" => {
                self.ensure_trima()?;
                let max_period = periods.iter().copied().map(|p| p as usize).max().unwrap_or(0);
                self.host_i32.clear();
                self.host_i32
                    .extend(periods.iter().map(|&p| first_valid as i32 + p - 1));
                Self::ensure_len_i32(&mut self.d_i32, n_combos)?;
                self.d_i32
                    .as_mut()
                    .unwrap()
                    .copy_from(&self.host_i32)
                    .map_err(|e| e.to_string())?;

                let trima = self.trima.as_ref().unwrap();
                trima.trima_batch_device(
                    d_prices,
                    d_periods,
                    self.d_i32.as_ref().unwrap(),
                    series_len,
                    n_combos,
                    max_period,
                    d_out,
                )
                .map_err(|e| e.to_string())?;
                trima.synchronize().map_err(|e| e.to_string())?;
                Ok(())
            }

            "sinwma" => {
                self.ensure_sinwma()?;
                let max_period = periods.iter().copied().map(|p| p as usize).max().unwrap_or(0);
                let sinwma = self.sinwma.as_ref().unwrap();
                sinwma
                    .sinwma_batch_device(
                        d_prices,
                        d_periods,
                        series_len as i32,
                        n_combos as i32,
                        first_valid as i32,
                        max_period as i32,
                        d_out,
                    )
                    .map_err(|e| e.to_string())?;
                sinwma.synchronize().map_err(|e| e.to_string())?;
                Ok(())
            }

            "epma" => {
                self.ensure_epma()?;
                let offset_f = Self::get_param_f64(params, "offset").unwrap_or(0.0).round();
                let offset_u = if offset_f < 0.0 { 0usize } else { offset_f as usize };
                let max_period = periods.iter().copied().map(|p| p as usize).max().unwrap_or(0);

                self.host_i32.clear();
                self.host_i32
                    .extend(periods.iter().map(|_p| offset_u as i32));
                Self::ensure_len_i32(&mut self.d_i32, n_combos)?;
                self.d_i32
                    .as_mut()
                    .unwrap()
                    .copy_from(&self.host_i32)
                    .map_err(|e| e.to_string())?;

                let epma = self.epma.as_ref().unwrap();
                epma.epma_batch_device(
                    d_prices,
                    d_periods,
                    self.d_i32.as_ref().unwrap(),
                    series_len,
                    n_combos,
                    first_valid,
                    max_period,
                    d_out,
                )
                .map_err(|e| e.to_string())?;
                epma.synchronize().map_err(|e| e.to_string())?;
                Ok(())
            }

            "wilders" => {
                self.ensure_wilders()?;
                self.host_i32.clear();
                self.host_i32
                    .extend(periods.iter().map(|&p| first_valid as i32 + p - 1));
                Self::ensure_len_i32(&mut self.d_i32, n_combos)?;
                self.d_i32
                    .as_mut()
                    .unwrap()
                    .copy_from(&self.host_i32)
                    .map_err(|e| e.to_string())?;

                self.host_f32.clear();
                self.host_f32
                    .extend(periods.iter().map(|&p| 1.0f32 / (p as f32)));
                Self::ensure_len_f32(&mut self.d_f32, n_combos)?;
                self.d_f32
                    .as_mut()
                    .unwrap()
                    .copy_from(&self.host_f32)
                    .map_err(|e| e.to_string())?;

                let wilders = self.wilders.as_ref().unwrap();
                wilders
                    .wilders_batch_device(
                        d_prices,
                        d_periods,
                        self.d_f32.as_ref().unwrap(),
                        self.d_i32.as_ref().unwrap(),
                        series_len as i32,
                        first_valid as i32,
                        n_combos as i32,
                        d_out,
                    )
                    .map_err(|e| e.to_string())?;
                wilders.synchronize().map_err(|e| e.to_string())?;
                Ok(())
            }

            "maaq" => {
                self.ensure_maaq()?;
                let max_period = periods.iter().copied().map(|p| p as usize).max().unwrap_or(0);
                let fast_p = Self::get_param_f64(params, "fast_period").unwrap_or(2.0).round();
                let slow_p = Self::get_param_f64(params, "slow_period").unwrap_or(30.0).round();
                let fast_u = fast_p.max(1.0) as usize;
                let slow_u = slow_p.max(1.0) as usize;
                let fast_sc = 2.0f32 / (fast_u as f32 + 1.0f32);
                let slow_sc = 2.0f32 / (slow_u as f32 + 1.0f32);

                self.host_f32.clear();
                self.host_f32.resize(n_combos, fast_sc);
                self.host_f32_b.clear();
                self.host_f32_b.resize(n_combos, slow_sc);

                Self::ensure_len_f32(&mut self.d_f32, n_combos)?;
                Self::ensure_len_f32(&mut self.d_f32_b, n_combos)?;
                self.d_f32
                    .as_mut()
                    .unwrap()
                    .copy_from(&self.host_f32)
                    .map_err(|e| e.to_string())?;
                self.d_f32_b
                    .as_mut()
                    .unwrap()
                    .copy_from(&self.host_f32_b)
                    .map_err(|e| e.to_string())?;

                let maaq = self.maaq.as_ref().unwrap();
                maaq.maaq_batch_device(
                    d_prices,
                    d_periods,
                    self.d_f32.as_ref().unwrap(),
                    self.d_f32_b.as_ref().unwrap(),
                    first_valid,
                    series_len,
                    n_combos,
                    max_period,
                    d_out,
                )
                .map_err(|e| e.to_string())?;
                maaq.synchronize().map_err(|e| e.to_string())?;
                Ok(())
            }

            "mwdx" => {
                self.ensure_mwdx()?;
                // mwdx factor = 2/(period+1)
                self.host_f32.clear();
                self.host_f32
                    .extend(periods.iter().map(|&p| 2.0f32 / (p as f32 + 1.0f32)));
                Self::ensure_len_f32(&mut self.d_f32, n_combos)?;
                self.d_f32
                    .as_mut()
                    .unwrap()
                    .copy_from(&self.host_f32)
                    .map_err(|e| e.to_string())?;

                let mwdx = self.mwdx.as_ref().unwrap();
                mwdx.mwdx_batch_device(
                    d_prices,
                    self.d_f32.as_ref().unwrap(),
                    series_len,
                    first_valid,
                    n_combos,
                    d_out,
                )
                .map_err(|e| e.to_string())?;
                mwdx.synchronize().map_err(|e| e.to_string())?;
                Ok(())
            }

            "cwma" => {
                self.ensure_cwma()?;
                let max_period = periods.iter().copied().map(|p| p as usize).max().unwrap_or(0);
                if max_period < 2 {
                    return Err("cwma max_period must be >= 2".to_string());
                }

                if let Some(cached) = self.cwma_cache.get(periods) {
                    let cwma = self.cwma.as_ref().unwrap();
                    cwma.cwma_batch_device(
                        d_prices,
                        &cached.d_weights,
                        d_periods,
                        &cached.d_ones,
                        cached.max_period as i32,
                        series_len as i32,
                        n_combos as i32,
                        first_valid as i32,
                        d_out,
                    )
                    .map_err(|e| e.to_string())?;
                    cwma.synchronize().map_err(|e| e.to_string())?;
                    return Ok(());
                }

                self.host_f32.clear();
                self.host_f32.resize(n_combos * max_period, 0.0f32);
                self.host_f32_b.clear();
                self.host_f32_b.resize(n_combos, 1.0f32);
                for (idx, &p_i32) in periods.iter().enumerate() {
                    let p = p_i32 as usize;
                    let wlen = p.saturating_sub(1);
                    let mut norm = 0.0f32;
                    let base = idx * max_period;
                    for k in 0..wlen {
                        let w = ((p - k) as f32).powi(3);
                        self.host_f32[base + k] = w;
                        norm += w;
                    }
                    let inv = 1.0f32 / norm.max(1e-20);
                    for k in 0..wlen {
                        self.host_f32[base + k] *= inv;
                    }
                }

                let d_weights = DeviceBuffer::from_slice(&self.host_f32).map_err(|e| e.to_string())?;
                let d_ones = DeviceBuffer::from_slice(&self.host_f32_b).map_err(|e| e.to_string())?;

                let cwma = self.cwma.as_ref().unwrap();
                cwma.cwma_batch_device(
                    d_prices,
                    &d_weights,
                    d_periods,
                    &d_ones,
                    max_period as i32,
                    series_len as i32,
                    n_combos as i32,
                    first_valid as i32,
                    d_out,
                )
                .map_err(|e| e.to_string())?;
                cwma.synchronize().map_err(|e| e.to_string())?;

                self.cwma_cache.insert(
                    periods.to_vec().into_boxed_slice(),
                    CwmaDeviceConsts {
                        d_weights,
                        d_ones,
                        max_period,
                    },
                );
                Ok(())
            }

            "cora_wave" => {
                self.ensure_cora_wave()?;
                let r_multi = Self::get_param_f64(params, "r_multi").unwrap_or(2.0);
                if !r_multi.is_finite() || r_multi < 0.0 {
                    return Err("cora_wave r_multi must be finite and >= 0".to_string());
                }
                let smooth = match Self::get_param_f64(params, "smooth") {
                    None => true,
                    Some(v) => {
                        let r = v.round();
                        if (v - r).abs() > 1e-9 {
                            return Err("cora_wave smooth must be an integer (0 or 1)".to_string());
                        }
                        match r as i32 {
                            0 => false,
                            1 => true,
                            _ => {
                                return Err("cora_wave smooth must be 0 or 1".to_string());
                            }
                        }
                    }
                };

                let key = CoraWaveParamKey {
                    r_multi_bits: r_multi.to_bits(),
                };
                if let Some(cached) = self
                    .cora_wave_cache
                    .get(&key)
                    .and_then(|m| m.get(periods))
                {
                    self.host_i32.clear();
                    self.host_i32.resize(n_combos, 1);
                    self.host_i32_b.clear();
                    self.host_i32_b.resize(n_combos, first_valid as i32);

                    for (idx, &p_i32) in periods.iter().enumerate() {
                        let p = p_i32 as usize;
                        if p == 0 {
                            return Err("cora_wave period must be > 0".to_string());
                        }
                        self.host_i32_b[idx] = (first_valid + p - 1) as i32;
                        let sp = if smooth {
                            ((p as f64).sqrt().round() as i32).max(1)
                        } else {
                            1
                        };
                        self.host_i32[idx] = sp;
                    }

                    let cora = self.cora_wave.as_ref().unwrap();
                    if smooth {
                        Self::ensure_len_i32(&mut self.d_i32, n_combos)?;
                        Self::ensure_len_i32(&mut self.d_i32_b, n_combos)?;
                        self.d_i32
                            .as_mut()
                            .unwrap()
                            .copy_from(&self.host_i32)
                            .map_err(|e| e.to_string())?;
                        self.d_i32_b
                            .as_mut()
                            .unwrap()
                            .copy_from(&self.host_i32_b)
                            .map_err(|e| e.to_string())?;

                        Self::ensure_len_f32(&mut self.cora_wave_tmp, n_combos * series_len)?;
                        cora.cora_wave_batch_device_into(
                            d_prices,
                            &cached.d_weights,
                            d_periods,
                            &cached.d_inv_norms,
                            cached.max_period,
                            series_len,
                            n_combos,
                            first_valid,
                            true,
                            Some(self.d_i32.as_ref().unwrap()),
                            Some(self.d_i32_b.as_ref().unwrap()),
                            Some(self.cora_wave_tmp.as_mut().unwrap()),
                            d_out,
                        )
                        .map_err(|e| e.to_string())?;
                    } else {
                        cora.cora_wave_batch_device_into(
                            d_prices,
                            &cached.d_weights,
                            d_periods,
                            &cached.d_inv_norms,
                            cached.max_period,
                            series_len,
                            n_combos,
                            first_valid,
                            false,
                            None,
                            None,
                            None,
                            d_out,
                        )
                        .map_err(|e| e.to_string())?;
                    }
                    cora.synchronize().map_err(|e| e.to_string())?;
                    return Ok(());
                }

                let max_period = periods.iter().copied().map(|p| p as usize).max().unwrap_or(0);
                if max_period == 0 {
                    return Err("cora_wave max_period is 0".to_string());
                }

                self.host_f32.clear();
                self.host_f32.resize(n_combos * max_period, 0.0f32);
                self.host_f32_b.clear();
                self.host_f32_b.resize(n_combos, 1.0f32);
                self.host_i32.clear();
                self.host_i32.resize(n_combos, 1);
                self.host_i32_b.clear();
                self.host_i32_b.resize(n_combos, first_valid as i32);

                for (idx, &p_i32) in periods.iter().enumerate() {
                    let p = p_i32 as usize;
                    if p == 0 {
                        return Err("cora_wave period must be > 0".to_string());
                    }
                    let base = idx * max_period;
                    if p == 1 {
                        self.host_f32[base] = 1.0f32;
                        self.host_f32_b[idx] = 1.0f32;
                    } else {
                        let start_wt = 0.01f64;
                        let end_wt = p as f64;
                        let r = (end_wt / start_wt).powf(1.0 / (p as f64 - 1.0)) - 1.0;
                        let base_w = 1.0 + r * r_multi;
                        let mut w = start_wt * base_w;
                        let mut sum = 0.0f64;
                        for j in 0..p {
                            self.host_f32[base + j] = w as f32;
                            sum += w;
                            w *= base_w;
                        }
                        self.host_f32_b[idx] = (1.0f64 / sum.max(1e-30)) as f32;
                    }

                    self.host_i32_b[idx] = (first_valid + p - 1) as i32;
                    let sp = if smooth {
                        ((p as f64).sqrt().round() as i32).max(1)
                    } else {
                        1
                    };
                    self.host_i32[idx] = sp;
                }

                let d_weights = DeviceBuffer::from_slice(&self.host_f32).map_err(|e| e.to_string())?;
                let d_inv_norms = DeviceBuffer::from_slice(&self.host_f32_b).map_err(|e| e.to_string())?;

                let cora = self.cora_wave.as_ref().unwrap();
                if smooth {
                    Self::ensure_len_i32(&mut self.d_i32, n_combos)?;
                    Self::ensure_len_i32(&mut self.d_i32_b, n_combos)?;
                    self.d_i32
                        .as_mut()
                        .unwrap()
                        .copy_from(&self.host_i32)
                        .map_err(|e| e.to_string())?;
                    self.d_i32_b
                        .as_mut()
                        .unwrap()
                        .copy_from(&self.host_i32_b)
                        .map_err(|e| e.to_string())?;

                    Self::ensure_len_f32(&mut self.cora_wave_tmp, n_combos * series_len)?;
                    cora.cora_wave_batch_device_into(
                        d_prices,
                        &d_weights,
                        d_periods,
                        &d_inv_norms,
                        max_period,
                        series_len,
                        n_combos,
                        first_valid,
                        true,
                        Some(self.d_i32.as_ref().unwrap()),
                        Some(self.d_i32_b.as_ref().unwrap()),
                        Some(self.cora_wave_tmp.as_mut().unwrap()),
                        d_out,
                    )
                    .map_err(|e| e.to_string())?;
                } else {
                    cora.cora_wave_batch_device_into(
                        d_prices,
                        &d_weights,
                        d_periods,
                        &d_inv_norms,
                        max_period,
                        series_len,
                        n_combos,
                        first_valid,
                        false,
                        None,
                        None,
                        None,
                        d_out,
                    )
                    .map_err(|e| e.to_string())?;
                }
                cora.synchronize().map_err(|e| e.to_string())?;

                self.cora_wave_cache
                    .entry(key)
                    .or_insert_with(HashMap::new)
                    .insert(
                        periods.to_vec().into_boxed_slice(),
                        CoraWaveDeviceConsts {
                            d_weights,
                            d_inv_norms,
                            max_period,
                        },
                    );
                Ok(())
            }

            "fwma" => {
                self.ensure_fwma()?;
                let max_period = periods.iter().copied().map(|p| p as usize).max().unwrap_or(0);
                if max_period == 0 {
                    return Err("fwma max_period is 0".to_string());
                }

                if let Some(cached) = self.fwma_cache.get(periods) {
                    self.host_i32.clear();
                    self.host_i32.reserve(n_combos);
                    for &p_i32 in periods {
                        let p = p_i32 as usize;
                        self.host_i32.push((first_valid + p - 1) as i32);
                    }
                    Self::ensure_len_i32(&mut self.d_i32, n_combos)?;
                    self.d_i32
                        .as_mut()
                        .unwrap()
                        .copy_from(&self.host_i32)
                        .map_err(|e| e.to_string())?;

                    let fwma = self.fwma.as_ref().unwrap();
                    fwma.fwma_batch_device(
                        d_prices,
                        &cached.d_weights,
                        d_periods,
                        self.d_i32.as_ref().unwrap(),
                        series_len,
                        n_combos,
                        cached.max_period,
                        d_out,
                    )
                    .map_err(|e| e.to_string())?;
                    fwma.synchronize().map_err(|e| e.to_string())?;
                    return Ok(());
                }

                self.host_f32.clear();
                self.host_f32.resize(n_combos * max_period, 0.0f32);
                self.host_i32.clear();
                self.host_i32.reserve(n_combos);

                for (idx, &p_i32) in periods.iter().enumerate() {
                    let p = p_i32 as usize;
                    let w = compute_fibonacci_weights_f32(p)?;
                    let base = idx * max_period;
                    self.host_f32[base..base + p].copy_from_slice(&w);
                    self.host_i32.push((first_valid + p - 1) as i32);
                }

                Self::ensure_len_i32(&mut self.d_i32, n_combos)?;
                self.d_i32
                    .as_mut()
                    .unwrap()
                    .copy_from(&self.host_i32)
                    .map_err(|e| e.to_string())?;

                let d_weights = DeviceBuffer::from_slice(&self.host_f32).map_err(|e| e.to_string())?;

                let fwma = self.fwma.as_ref().unwrap();
                fwma.fwma_batch_device(
                    d_prices,
                    &d_weights,
                    d_periods,
                    self.d_i32.as_ref().unwrap(),
                    series_len,
                    n_combos,
                    max_period,
                    d_out,
                )
                .map_err(|e| e.to_string())?;
                fwma.synchronize().map_err(|e| e.to_string())?;

                self.fwma_cache.insert(
                    periods.to_vec().into_boxed_slice(),
                    FwmaDeviceConsts { d_weights, max_period },
                );
                Ok(())
            }

            "pwma" => {
                self.ensure_pwma()?;
                let max_period = periods.iter().copied().map(|p| p as usize).max().unwrap_or(0);
                if max_period == 0 {
                    return Err("pwma max_period is 0".to_string());
                }

                if let Some(cached) = self.pwma_cache.get(periods) {
                    self.host_i32.clear();
                    self.host_i32.reserve(n_combos);
                    for &p_i32 in periods {
                        let p = p_i32 as usize;
                        self.host_i32.push((first_valid + p - 1) as i32);
                    }
                    Self::ensure_len_i32(&mut self.d_i32, n_combos)?;
                    self.d_i32
                        .as_mut()
                        .unwrap()
                        .copy_from(&self.host_i32)
                        .map_err(|e| e.to_string())?;

                    let pwma = self.pwma.as_ref().unwrap();
                    pwma.pwma_batch_device(
                        d_prices,
                        &cached.d_weights,
                        d_periods,
                        self.d_i32.as_ref().unwrap(),
                        series_len,
                        n_combos,
                        cached.max_period,
                        d_out,
                    )
                    .map_err(|e| e.to_string())?;
                    pwma.synchronize().map_err(|e| e.to_string())?;
                    return Ok(());
                }

                self.host_f32.clear();
                self.host_f32.resize(n_combos * max_period, 0.0f32);
                self.host_i32.clear();
                self.host_i32.reserve(n_combos);

                for (idx, &p_i32) in periods.iter().enumerate() {
                    let p = p_i32 as usize;
                    let w = compute_pascal_weights_f32(p)?;
                    let base = idx * max_period;
                    self.host_f32[base..base + p].copy_from_slice(&w);
                    self.host_i32.push((first_valid + p - 1) as i32);
                }

                Self::ensure_len_i32(&mut self.d_i32, n_combos)?;
                self.d_i32
                    .as_mut()
                    .unwrap()
                    .copy_from(&self.host_i32)
                    .map_err(|e| e.to_string())?;

                let d_weights = DeviceBuffer::from_slice(&self.host_f32).map_err(|e| e.to_string())?;

                let pwma = self.pwma.as_ref().unwrap();
                pwma.pwma_batch_device(
                    d_prices,
                    &d_weights,
                    d_periods,
                    self.d_i32.as_ref().unwrap(),
                    series_len,
                    n_combos,
                    max_period,
                    d_out,
                )
                .map_err(|e| e.to_string())?;
                pwma.synchronize().map_err(|e| e.to_string())?;

                self.pwma_cache.insert(
                    periods.to_vec().into_boxed_slice(),
                    PwmaDeviceConsts { d_weights, max_period },
                );
                Ok(())
            }

            "srwma" => {
                self.ensure_srwma()?;
                let max_period = periods.iter().copied().map(|p| p as usize).max().unwrap_or(0);
                if max_period < 2 {
                    return Err("srwma max_period must be >= 2".to_string());
                }
                let max_wlen = max_period - 1;

                if let Some(cached) = self.srwma_cache.get(periods) {
                    self.host_i32.clear();
                    self.host_i32.reserve(n_combos);
                    for &p_i32 in periods {
                        let p = p_i32 as usize;
                        self.host_i32.push((first_valid + p + 1) as i32);
                    }

                    Self::ensure_len_i32(&mut self.d_i32, n_combos)?;
                    self.d_i32
                        .as_mut()
                        .unwrap()
                        .copy_from(&self.host_i32)
                        .map_err(|e| e.to_string())?;

                    let srwma = self.srwma.as_ref().unwrap();
                    srwma.srwma_batch_device(
                        d_prices,
                        &cached.d_weights,
                        d_periods,
                        self.d_i32.as_ref().unwrap(),
                        &cached.d_inv_norms,
                        series_len,
                        first_valid,
                        cached.max_wlen,
                        n_combos,
                        d_out,
                    )
                    .map_err(|e| e.to_string())?;
                    srwma.synchronize().map_err(|e| e.to_string())?;
                    return Ok(());
                }

                self.host_f32.clear();
                self.host_f32.resize(n_combos * max_wlen, 0.0f32);
                self.host_f32_b.clear();
                self.host_f32_b.resize(n_combos, 1.0f32);
                self.host_i32.clear();
                self.host_i32.reserve(n_combos);

                for (idx, &p_i32) in periods.iter().enumerate() {
                    let p = p_i32 as usize;
                    let wlen = p - 1;
                    let base = idx * max_wlen;
                    let mut norm = 0.0f32;
                    for k in 0..wlen {
                        let w = ((p - k) as f32).sqrt();
                        self.host_f32[base + k] = w;
                        norm += w;
                    }
                    self.host_f32_b[idx] = 1.0f32 / norm.max(1e-20);
                    self.host_i32.push((first_valid + p + 1) as i32);
                }

                Self::ensure_len_i32(&mut self.d_i32, n_combos)?;
                self.d_i32
                    .as_mut()
                    .unwrap()
                    .copy_from(&self.host_i32)
                    .map_err(|e| e.to_string())?;

                let d_weights = DeviceBuffer::from_slice(&self.host_f32).map_err(|e| e.to_string())?;
                let d_inv_norms = DeviceBuffer::from_slice(&self.host_f32_b).map_err(|e| e.to_string())?;

                let srwma = self.srwma.as_ref().unwrap();
                srwma.srwma_batch_device(
                    d_prices,
                    &d_weights,
                    d_periods,
                    self.d_i32.as_ref().unwrap(),
                    &d_inv_norms,
                    series_len,
                    first_valid,
                    max_wlen,
                    n_combos,
                    d_out,
                )
                .map_err(|e| e.to_string())?;
                srwma.synchronize().map_err(|e| e.to_string())?;

                self.srwma_cache.insert(
                    periods.to_vec().into_boxed_slice(),
                    SrwmaDeviceConsts {
                        d_weights,
                        d_inv_norms,
                        max_wlen,
                    },
                );
                Ok(())
            }

            "supersmoother_3_pole" => {
                self.ensure_supersmoother_3_pole()?;
                let ss = self.supersmoother_3_pole.as_ref().unwrap();
                ss.supersmoother_3_pole_batch_device(
                    d_prices,
                    d_periods,
                    series_len,
                    n_combos,
                    first_valid,
                    d_out,
                )
                .map_err(|e| e.to_string())?;
                ss.synchronize().map_err(|e| e.to_string())?;
                Ok(())
            }

            "supersmoother" => {
                self.ensure_supersmoother()?;
                let ss = self.supersmoother.as_ref().unwrap();
                ss.supersmoother_batch_device(
                    d_prices,
                    d_periods,
                    series_len,
                    n_combos,
                    first_valid,
                    d_out,
                )
                .map_err(|e| e.to_string())?;
                ss.synchronize().map_err(|e| e.to_string())?;
                Ok(())
            }

            "zlema" => {
                self.ensure_zlema()?;
                self.host_i32.clear();
                self.host_f32.clear();
                self.host_i32.reserve(n_combos);
                self.host_f32.reserve(n_combos);

                let mut max_lag: i32 = 0;
                for &p_i32 in periods {
                    let p = p_i32 as usize;
                    let lag = ((p.saturating_sub(1)) / 2) as i32;
                    max_lag = max_lag.max(lag);
                    self.host_i32.push(lag);
                    self.host_f32.push(2.0f32 / (p as f32 + 1.0f32));
                }

                Self::ensure_len_i32(&mut self.d_i32, n_combos)?;
                Self::ensure_len_f32(&mut self.d_f32, n_combos)?;
                self.d_i32
                    .as_mut()
                    .unwrap()
                    .copy_from(&self.host_i32)
                    .map_err(|e| e.to_string())?;
                self.d_f32
                    .as_mut()
                    .unwrap()
                    .copy_from(&self.host_f32)
                    .map_err(|e| e.to_string())?;

                let zlema = self.zlema.as_ref().unwrap();
                zlema.zlema_batch_device(
                    d_prices,
                    d_periods,
                    self.d_i32.as_ref().unwrap(),
                    self.d_f32.as_ref().unwrap(),
                    series_len,
                    first_valid,
                    n_combos,
                    max_lag,
                    d_out,
                )
                .map_err(|e| e.to_string())?;
                zlema.synchronize().map_err(|e| e.to_string())?;
                Ok(())
            }

            "nma" => {
                self.ensure_nma()?;
                let max_period = periods.iter().copied().map(|p| p as usize).max().unwrap_or(0);
                if max_period == 0 {
                    return Err("nma max_period is 0".to_string());
                }

                Self::ensure_len_f32(&mut self.nma_abs_diffs, series_len)?;

                let nma = self.nma.as_ref().unwrap();
                {
                    let diffs = self.nma_abs_diffs.as_mut().unwrap();
                    nma.nma_abs_log_diffs_f32_device(d_prices, series_len, first_valid, diffs)
                        .map_err(|e| e.to_string())?;
                }
                nma.nma_batch_device(
                    d_prices,
                    self.nma_abs_diffs.as_ref().unwrap(),
                    d_periods,
                    series_len,
                    n_combos,
                    first_valid,
                    max_period,
                    d_out,
                )
                .map_err(|e| e.to_string())?;
                nma.synchronize().map_err(|e| e.to_string())?;
                Ok(())
            }

            "hma" => {
                self.ensure_hma()?;
                let mut max_sqrt_len: usize = 0;
                for &p_i32 in periods {
                    let p = p_i32 as usize;
                    let sqrt_len = ((p as f64).sqrt().floor() as usize).max(1);
                    max_sqrt_len = max_sqrt_len.max(sqrt_len);
                }
                if max_sqrt_len == 0 {
                    return Err("hma max_sqrt_len is 0".to_string());
                }
                let ring_elems = n_combos
                    .checked_mul(max_sqrt_len)
                    .ok_or_else(|| "hma ring_elems overflow".to_string())?;
                Self::ensure_len_f32(&mut self.hma_ring, ring_elems)?;

                let hma = self.hma.as_ref().unwrap();
                hma.hma_batch_device(
                    d_prices,
                    d_periods,
                    series_len,
                    n_combos,
                    first_valid,
                    max_sqrt_len,
                    self.hma_ring.as_mut().unwrap(),
                    d_out,
                )
                .map_err(|e| e.to_string())?;
                hma.synchronize().map_err(|e| e.to_string())?;
                Ok(())
            }

            "jma" => {
                self.ensure_jma()?;
                let phase = Self::get_param_f64(params, "phase").unwrap_or(50.0);
                let power_f = Self::get_param_f64(params, "power").unwrap_or(2.0).round();
                let power_u: u32 = if power_f < 1.0 { 1 } else { power_f as u32 };
                let phase_ratio: f32 = if phase < -100.0 {
                    0.5
                } else if phase > 100.0 {
                    2.5
                } else {
                    (phase / 100.0 + 1.5) as f32
                };

                self.host_f32.clear();
                self.host_f32_b.clear();
                self.host_f32_c.clear();
                self.host_f32.reserve(n_combos);
                self.host_f32_b.reserve(n_combos);
                self.host_f32_c.reserve(n_combos);

                for &p_i32 in periods {
                    let period = p_i32 as usize;
                    if period == 0 {
                        return Err("jma period is 0".to_string());
                    }

                    let numerator = 0.45f64 * (period as f64 - 1.0);
                    let denominator = numerator + 2.0;
                    if denominator.abs() < f64::EPSILON {
                        return Err("jma denominator is ~0".to_string());
                    }
                    let beta = numerator / denominator;
                    let alpha = beta.powi(power_u as i32);
                    let one_minus_beta = 1.0 - beta;

                    self.host_f32.push(alpha as f32);
                    self.host_f32_b.push(one_minus_beta as f32);
                    self.host_f32_c.push(phase_ratio);
                }

                Self::ensure_len_f32(&mut self.d_f32, n_combos)?;
                Self::ensure_len_f32(&mut self.d_f32_b, n_combos)?;
                Self::ensure_len_f32(&mut self.d_f32_c, n_combos)?;
                self.d_f32
                    .as_mut()
                    .unwrap()
                    .copy_from(&self.host_f32)
                    .map_err(|e| e.to_string())?;
                self.d_f32_b
                    .as_mut()
                    .unwrap()
                    .copy_from(&self.host_f32_b)
                    .map_err(|e| e.to_string())?;
                self.d_f32_c
                    .as_mut()
                    .unwrap()
                    .copy_from(&self.host_f32_c)
                    .map_err(|e| e.to_string())?;

                let jma = self.jma.as_ref().unwrap();
                jma.jma_batch_device(
                    d_prices,
                    self.d_f32.as_ref().unwrap(),
                    self.d_f32_b.as_ref().unwrap(),
                    self.d_f32_c.as_ref().unwrap(),
                    series_len,
                    n_combos,
                    first_valid,
                    d_out,
                )
                .map_err(|e| e.to_string())?;
                jma.synchronize().map_err(|e| e.to_string())?;
                Ok(())
            }

            "edcf" => {
                self.ensure_edcf()?;
                Self::ensure_len_f32(&mut self.edcf_dist, series_len)?;

                let combos: Vec<crate::indicators::moving_averages::EdcfParams> = periods
                    .iter()
                    .map(|&p| crate::indicators::moving_averages::EdcfParams {
                        period: Some(p as usize),
                    })
                    .collect();

                let edcf = self.edcf.as_ref().unwrap();
                edcf.edcf_batch_device(
                    d_prices,
                    &combos,
                    first_valid,
                    series_len,
                    self.edcf_dist.as_mut().unwrap(),
                    d_out,
                )
                .map_err(|e| e.to_string())?;
                Ok(())
            }

            "ehlers_itrend" => {
                self.ensure_ehlers_itrend()?;
                let warmup_f = Self::get_param_f64(params, "warmup_bars")
                    .unwrap_or(20.0)
                    .round();
                let warmup_u: usize = if warmup_f < 1.0 { 1 } else { warmup_f as usize };
                let max_dc = periods.iter().copied().map(|p| p as usize).max().unwrap_or(0);
                if max_dc == 0 {
                    return Err("ehlers_itrend max_dc is 0".to_string());
                }

                self.host_i32.clear();
                self.host_i32.resize(n_combos, warmup_u as i32);
                Self::ensure_len_i32(&mut self.d_i32, n_combos)?;
                self.d_i32
                    .as_mut()
                    .unwrap()
                    .copy_from(&self.host_i32)
                    .map_err(|e| e.to_string())?;

                let eit = self.ehlers_itrend.as_ref().unwrap();
                eit.ehlers_itrend_batch_device(
                    d_prices,
                    self.d_i32.as_ref().unwrap(),
                    d_periods,
                    series_len,
                    first_valid,
                    n_combos,
                    max_dc,
                    d_out,
                )
                .map_err(|e| e.to_string())?;
                eit.synchronize().map_err(|e| e.to_string())?;
                Ok(())
            }

            "vwma" => {
                self.ensure_vwma()?;
                let pv_prefix = self
                    .vwma_pv_prefix_f64
                    .as_ref()
                    .ok_or_else(|| {
                        "vwma prefixes missing (call ensure_vwma_prefix_pv_vol_f64 first)"
                            .to_string()
                    })?;
                let vol_prefix = self
                    .vwma_vol_prefix_f64
                    .as_ref()
                    .ok_or_else(|| {
                        "vwma prefixes missing (call ensure_vwma_prefix_pv_vol_f64 first)"
                            .to_string()
                    })?;
                let vwma = self.vwma.as_ref().unwrap();
                vwma.vwma_batch_device(
                    pv_prefix,
                    vol_prefix,
                    d_periods,
                    series_len,
                    n_combos,
                    first_valid,
                    d_out,
                )
                .map_err(|e| e.to_string())?;
                vwma.synchronize().map_err(|e| e.to_string())?;
                Ok(())
            }

            "vpwma" => {
                self.ensure_vpwma()?;
                let power = Self::get_param_f64(params, "power").unwrap_or(0.382);
                if !power.is_finite() {
                    return Err("vpwma power must be finite".to_string());
                }

                let key = VpwmaParamKey {
                    power_bits: power.to_bits(),
                };
                if let Some(cached) = self
                    .vpwma_cache
                    .get(&key)
                    .and_then(|m| m.get(periods))
                {
                    self.host_i32.clear();
                    self.host_i32.resize(n_combos, 0);
                    for (ci, &p_i) in periods.iter().enumerate() {
                        let period = p_i as i64;
                        if period < 2 {
                            return Err("vpwma period must be >= 2".to_string());
                        }
                        self.host_i32[ci] = (period as usize - 1) as i32;
                    }

                    Self::ensure_len_i32(&mut self.d_i32, n_combos)?;
                    self.d_i32
                        .as_mut()
                        .unwrap()
                        .copy_from(&self.host_i32)
                        .map_err(|e| e.to_string())?;

                    let vpwma = self.vpwma.as_ref().unwrap();
                    vpwma.vpwma_batch_device(
                        d_prices,
                        d_periods,
                        self.d_i32.as_ref().unwrap(),
                        &cached.d_weights,
                        &cached.d_inv_norms,
                        series_len,
                        cached.stride,
                        first_valid,
                        n_combos,
                        d_out,
                    )
                    .map_err(|e| e.to_string())?;
                    vpwma.synchronize().map_err(|e| e.to_string())?;
                    return Ok(());
                }

                let max_period = periods
                    .iter()
                    .copied()
                    .map(|p| p.max(0) as usize)
                    .max()
                    .unwrap_or(0);
                if max_period < 2 {
                    return Err("vpwma period must be >= 2".to_string());
                }
                let stride = max_period - 1;
                if stride == 0 {
                    return Err("vpwma stride must be > 0".to_string());
                }

                self.host_i32.clear();
                self.host_i32.resize(n_combos, 0);
                self.host_f32.clear();
                let weights_len = n_combos
                    .checked_mul(stride)
                    .ok_or_else(|| "vpwma weights size overflow".to_string())?;
                self.host_f32.resize(weights_len, 0.0);
                self.host_f32_b.clear();
                self.host_f32_b.resize(n_combos, 0.0);

                for (ci, &p_i) in periods.iter().enumerate() {
                    let period = p_i as i64;
                    if period < 2 {
                        return Err("vpwma period must be >= 2".to_string());
                    }
                    let period_u = period as usize;
                    let win_len = period_u - 1;
                    self.host_i32[ci] = win_len as i32;

                    let base = ci
                        .checked_mul(stride)
                        .ok_or_else(|| "vpwma weights index overflow".to_string())?;

                    let mut norm = 0.0f64;
                    for k in 0..win_len {
                        let w = (period_u as f64 - k as f64).powf(power);
                        self.host_f32[base + k] = w as f32;
                        norm += w;
                    }
                    if !norm.is_finite() || norm == 0.0 {
                        return Err(format!(
                            "vpwma invalid normalization for period {} power {}",
                            period_u, power
                        ));
                    }
                    self.host_f32_b[ci] = (1.0 / norm) as f32;
                }

                Self::ensure_len_i32(&mut self.d_i32, n_combos)?;
                self.d_i32
                    .as_mut()
                    .unwrap()
                    .copy_from(&self.host_i32)
                    .map_err(|e| e.to_string())?;

                let d_weights = DeviceBuffer::from_slice(&self.host_f32).map_err(|e| e.to_string())?;
                let d_inv_norms = DeviceBuffer::from_slice(&self.host_f32_b).map_err(|e| e.to_string())?;

                let vpwma = self.vpwma.as_ref().unwrap();
                vpwma.vpwma_batch_device(
                    d_prices,
                    d_periods,
                    self.d_i32.as_ref().unwrap(),
                    &d_weights,
                    &d_inv_norms,
                    series_len,
                    stride,
                    first_valid,
                    n_combos,
                    d_out,
                )
                .map_err(|e| e.to_string())?;
                vpwma.synchronize().map_err(|e| e.to_string())?;

                self.vpwma_cache
                    .entry(key)
                    .or_insert_with(HashMap::new)
                    .insert(
                        periods.to_vec().into_boxed_slice(),
                        VpwmaDeviceConsts {
                            d_weights,
                            d_inv_norms,
                            stride,
                        },
                    );
                Ok(())
            }

            "frama" => {
                self.ensure_frama()?;
                let d_high = inputs
                    .high
                    .ok_or_else(|| "frama requires inputs.high".to_string())?;
                let d_low = inputs
                    .low
                    .ok_or_else(|| "frama requires inputs.low".to_string())?;

                if d_high.len() != series_len || d_low.len() != series_len {
                    return Err("frama device input length mismatch".to_string());
                }

                let sc = Self::get_param_f64(params, "sc")
                    .unwrap_or(300.0)
                    .round() as i32;
                let fc = Self::get_param_f64(params, "fc")
                    .unwrap_or(1.0)
                    .round() as i32;
                if sc <= 0 || fc <= 0 {
                    return Err("frama sc/fc must be positive".to_string());
                }

                for &p in periods {
                    let window = p as i64;
                    if window <= 0 {
                        return Err("frama window must be > 0".to_string());
                    }
                    let win_u = window as usize;
                    let even = if win_u & 1 == 1 { win_u + 1 } else { win_u };
                    if even > 1024 {
                        return Err("frama evenized window exceeds CUDA limit".to_string());
                    }
                }

                self.host_i32.clear();
                self.host_i32.resize(n_combos, sc);
                self.host_i32_b.clear();
                self.host_i32_b.resize(n_combos, fc);
                Self::ensure_len_i32(&mut self.d_i32, n_combos)?;
                Self::ensure_len_i32(&mut self.d_i32_b, n_combos)?;
                self.d_i32
                    .as_mut()
                    .unwrap()
                    .copy_from(&self.host_i32)
                    .map_err(|e| e.to_string())?;
                self.d_i32_b
                    .as_mut()
                    .unwrap()
                    .copy_from(&self.host_i32_b)
                    .map_err(|e| e.to_string())?;

                let frama = self.frama.as_ref().unwrap();
                frama
                    .frama_batch_device(
                        d_high,
                        d_low,
                        inputs.close,
                        d_periods,
                        self.d_i32.as_ref().unwrap(),
                        self.d_i32_b.as_ref().unwrap(),
                        series_len,
                        first_valid,
                        n_combos,
                        d_out,
                    )
                    .map_err(|e| e.to_string())?;
                Ok(())
            }

            other => Err(format!("unsupported MA for VRAM kernel: {other}")),
        }
    }

    pub fn compute_period_ma_tm_into(
        &mut self,
        ma_type: &str,
        params: Option<&HashMap<String, f64>>,
        inputs: &VramMaInputs,
        series_len: usize,
        first_valid: usize,
        periods: &[i32],
        d_periods: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), String> {
        let ma = ma_type.trim().to_ascii_lowercase();
        let n_combos = periods.len();
        if n_combos == 0 {
            return Err("empty period list".to_string());
        }
        if d_periods.len() < n_combos {
            return Err("device periods buffer too small".to_string());
        }
        if d_out_tm.len() < n_combos.saturating_mul(series_len) {
            return Err("device output buffer too small".to_string());
        }
        if inputs.prices.len() != series_len {
            return Err("device prices buffer length mismatch".to_string());
        }
        if inputs.close.len() != series_len {
            return Err("device close buffer length mismatch".to_string());
        }
        let d_prices = inputs.prices;

        match ma.as_str() {
            "sma" => {
                self.ensure_sma()?;
                let prefix = self
                    .sma_prefix_f64
                    .as_ref()
                    .ok_or_else(|| "sma prefix missing (call ensure_sma_prefix_f64 first)".to_string())?;
                let sma = self.sma.as_ref().unwrap();
                sma.sma_batch_from_prefix_f64_device_into_tm(
                    prefix,
                    d_periods,
                    series_len,
                    n_combos,
                    first_valid,
                    d_out_tm,
                )
                .map_err(|e| e.to_string())?;
                sma.synchronize().map_err(|e| e.to_string())?;
                Ok(())
            }

            "alma" => {
                self.ensure_alma()?;
                let offset = Self::get_param_f64(params, "offset").unwrap_or(0.85);
                let sigma = Self::get_param_f64(params, "sigma").unwrap_or(6.0);
                let key = AlmaParamKey {
                    offset_bits: offset.to_bits(),
                    sigma_bits: sigma.to_bits(),
                };
                if let Some(cached) = self
                    .alma_cache
                    .get(&key)
                    .and_then(|m| m.get(periods))
                {
                    let alma = self.alma.as_ref().unwrap();
                    alma.alma_batch_device_tm(
                        d_prices,
                        &cached.d_weights,
                        d_periods,
                        &cached.d_inv_norms,
                        cached.max_period as i32,
                        series_len as i32,
                        n_combos as i32,
                        first_valid as i32,
                        d_out_tm,
                    )
                    .map_err(|e| e.to_string())?;
                    alma.synchronize().map_err(|e| e.to_string())?;
                    return Ok(());
                }

                let max_period = periods.iter().copied().map(|p| p as usize).max().unwrap_or(0);
                if max_period == 0 {
                    return Err("alma max_period is 0".to_string());
                }

                self.host_f32.clear();
                self.host_f32.resize(n_combos * max_period, 0.0f32);
                self.host_f32_b.clear();
                self.host_f32_b.resize(n_combos, 0.0f32);
                for (combo_idx, &p_i32) in periods.iter().enumerate() {
                    let period = p_i32 as usize;
                    let (mut w, inv_norm) = compute_alma_weights_f32(period, offset, sigma);
                    if inv_norm != 0.0 {
                        for wi in w.iter_mut() {
                            *wi *= inv_norm;
                        }
                    }
                    self.host_f32_b[combo_idx] = 1.0f32;
                    let base = combo_idx * max_period;
                    self.host_f32[base..base + period].copy_from_slice(&w);
                }

                let d_weights = DeviceBuffer::from_slice(&self.host_f32).map_err(|e| e.to_string())?;
                let d_inv_norms = DeviceBuffer::from_slice(&self.host_f32_b).map_err(|e| e.to_string())?;

                let alma = self.alma.as_ref().unwrap();
                alma.alma_batch_device_tm(
                    d_prices,
                    &d_weights,
                    d_periods,
                    &d_inv_norms,
                    max_period as i32,
                    series_len as i32,
                    n_combos as i32,
                    first_valid as i32,
                    d_out_tm,
                )
                .map_err(|e| e.to_string())?;
                alma.synchronize().map_err(|e| e.to_string())?;

                self.alma_cache
                    .entry(key)
                    .or_insert_with(HashMap::new)
                    .insert(
                        periods.to_vec().into_boxed_slice(),
                        AlmaDeviceConsts {
                            d_weights,
                            d_inv_norms,
                            max_period,
                        },
                    );
                Ok(())
            }

            other => Err(format!("unsupported MA for VRAM kernel TM output: {other}")),
        }
    }
}

fn compute_alma_weights_f32(period: usize, offset: f64, sigma: f64) -> (Vec<f32>, f32) {
    let m = (offset * (period as f64 - 1.0)) as f32;
    let s = (period as f64 / sigma) as f32;
    let s2 = 2.0f32 * s * s;
    let mut w = vec![0.0f32; period];
    let mut norm = 0.0f32;
    for i in 0..period {
        let diff = i as f32 - m;
        let wi = (-(diff * diff) / s2).exp();
        w[i] = wi;
        norm += wi;
    }
    (w, if norm != 0.0 { 1.0f32 / norm } else { 0.0f32 })
}

fn compute_pascal_weights_f32(period: usize) -> Result<Vec<f32>, String> {
    if period == 0 {
        return Err("pwma period must be > 0".to_string());
    }
    let n = period - 1;
    let mut row = Vec::with_capacity(period);
    let mut sum = 0.0f64;
    for r in 0..=n {
        let mut val = 1.0f64;
        for i in 0..r {
            val *= (n - i) as f64;
            val /= (i + 1) as f64;
        }
        row.push(val);
        sum += val;
    }
    if sum == 0.0 {
        return Err(format!("pwma Pascal weights sum to zero for period {period}"));
    }
    let inv = 1.0 / sum;
    Ok(row.into_iter().map(|v| (v * inv) as f32).collect())
}

fn compute_fibonacci_weights_f32(period: usize) -> Result<Vec<f32>, String> {
    if period == 0 {
        return Err("fwma period must be > 0".to_string());
    }
    if period == 1 {
        return Ok(vec![1.0f32]);
    }
    let mut fib = vec![1.0f64; period];
    for i in 2..period {
        fib[i] = fib[i - 1] + fib[i - 2];
    }
    let sum: f64 = fib.iter().sum();
    if sum == 0.0 {
        return Err(format!("fwma Fibonacci weights sum to zero for period {period}"));
    }
    let inv = 1.0 / sum;
    Ok(fib.into_iter().map(|v| (v * inv) as f32).collect())
}
