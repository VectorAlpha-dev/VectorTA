use rayon::prelude::*;
use anyhow::Result;
use crate::backends::types::{OptimizeRequest, OptimizeResponse, OptimizeResponseMeta, AxisMeta, expand_range};
use serde_json::Value as JsonValue;
use my_project::indicators::moving_averages::alma::AlmaBuilder;
use my_project::indicators::moving_averages::dma::DmaBuilder;
use my_project::indicators::moving_averages::dema::DemaBuilder;
use my_project::indicators::moving_averages::cwma::CwmaBuilder;
use my_project::indicators::moving_averages::edcf::EdcfBuilder;
use my_project::indicators::moving_averages::ehma::EhmaBuilder;
use my_project::indicators::moving_averages::ema::EmaBuilder;
use my_project::indicators::moving_averages::epma::EpmaBuilder;
use my_project::indicators::moving_averages::fwma::FwmaBuilder;
use my_project::indicators::moving_averages::gaussian::GaussianBuilder;
use my_project::indicators::moving_averages::highpass::HighPassBuilder;
use my_project::indicators::moving_averages::highpass_2_pole::HighPass2Builder;
use my_project::indicators::moving_averages::hma::HmaBuilder;
use my_project::indicators::moving_averages::hwma::HwmaBuilder;
use my_project::indicators::moving_averages::jma::JmaBuilder;
use my_project::indicators::moving_averages::jsa::JsaBuilder;
use my_project::indicators::moving_averages::kama::KamaBuilder;
use my_project::indicators::moving_averages::linreg::LinRegBuilder;
use my_project::indicators::moving_averages::nama::NamaBuilder;
use my_project::indicators::moving_averages::nma::NmaBuilder;
use my_project::indicators::moving_averages::pwma::PwmaBuilder;
use my_project::indicators::moving_averages::reflex::ReflexBuilder;
use my_project::indicators::moving_averages::sama::SamaBuilder;
use my_project::indicators::moving_averages::sinwma::SinWmaBuilder;
use my_project::indicators::moving_averages::sma::SmaBuilder;
use my_project::indicators::moving_averages::smma::SmmaBuilder;
use my_project::indicators::moving_averages::sqwma::SqwmaBuilder;
use my_project::indicators::moving_averages::srwma::SrwmaBuilder;
use my_project::indicators::moving_averages::supersmoother::SuperSmootherBuilder;
use my_project::indicators::moving_averages::supersmoother_3_pole::SuperSmoother3PoleBuilder;
use my_project::indicators::moving_averages::tema::TemaBuilder;
use my_project::indicators::moving_averages::trima::TrimaBuilder;
use my_project::indicators::moving_averages::swma::SwmaBuilder;
use my_project::indicators::moving_averages::tilson::TilsonBuilder;
use my_project::indicators::moving_averages::tradjema::TradjemaBuilder;
use my_project::indicators::moving_averages::trendflex::TrendFlexBuilder;
use my_project::indicators::moving_averages::uma::UmaBuilder;
use my_project::indicators::moving_averages::volatility_adjusted_ma::VamaBuilder;
use my_project::indicators::moving_averages::volume_adjusted_ma::VolumeAdjustedMaBuilder;
use my_project::indicators::moving_averages::vpwma::VpwmaBuilder;
use my_project::indicators::moving_averages::vwma::VwmaBuilder;
use my_project::indicators::moving_averages::wilders::WildersBuilder;
use my_project::indicators::moving_averages::wma::WmaBuilder;
use my_project::indicators::moving_averages::zlema::ZlemaBuilder;
use my_project::utilities::enums::Kernel;

fn gen_synthetic(len: usize) -> Vec<f64> {
    let mut v = vec![f64::NAN; len];
    for i in 3..len { let x = i as f64; v[i] = (x * 0.001).sin() + 0.0001 * x; }
    v
}

#[inline]
fn backtest_pair(
    prices: &[f64],
    fast: &[f64],
    slow: &[f64],
    period_f: usize,
    period_s: usize,
    commission: f64,
) -> [f32; 5] {
    let first = prices.iter().position(|x| !x.is_nan()).unwrap_or(0);
    let t0 = first + period_f.max(period_s) - 1;
    if t0 + 1 >= prices.len() { return [0.0; 5]; }
    let mut equity = 1.0_f64;
    let mut peak = 1.0_f64;
    let mut max_dd = 0.0_f64;
    let mut trades = 0_i32;
    let mut pos = if fast[t0] > slow[t0] {1} else if fast[t0] < slow[t0] {-1} else {0};
    let mut mean = 0.0_f64; let mut m2 = 0.0_f64; let mut n = 0_i64;
    for t in (t0+1)..prices.len() {
        let f_prev = fast[t-1]; let s_prev = slow[t-1];
        let f_cur = fast[t];   let s_cur = slow[t];
        let sign_prev = if f_prev > s_prev {1} else if f_prev < s_prev {-1} else {0};
        let sign_cur  = if f_cur  > s_cur  {1} else if f_cur  < s_cur  {-1} else {0};
        if sign_cur != pos {
            if pos == 0 && sign_cur != 0 { if commission>0.0 { equity *= 1.0-commission; } trades += 1; pos = sign_cur; }
            else if pos != 0 && sign_cur == 0 { if commission>0.0 { equity *= 1.0-commission; } trades += 1; pos = 0; }
            else if pos != 0 && sign_cur != 0 { if commission>0.0 { equity *= (1.0-commission)*(1.0-commission); } trades += 2; pos = sign_cur; }
        }
        let ret_t = prices[t] / prices[t-1] - 1.0;
        let strat = if pos==0 {0.0} else {(pos as f64) * ret_t};
        equity *= 1.0 + strat;
        if equity > peak { peak = equity; }
        let dd = if peak>0.0 { (peak - equity)/peak } else {0.0};
        if dd > max_dd { max_dd = dd; }
        n += 1; let delta = strat - mean; mean += delta / (n as f64); m2 += delta * (strat - mean);
    }
    let variance = if n>1 { m2 / ((n-1) as f64) } else { 0.0 };
    let std = if variance>0.0 { variance.sqrt() } else { 0.0 };
    [
        (equity - 1.0) as f32,
        trades as f32,
        max_dd as f32,
        mean as f32,
        std as f32,
    ]
}

use my_project::indicators::moving_averages::frama::FramaBuilder;
use my_project::indicators::moving_averages::vwap::VwapBuilder;
use my_project::indicators::moving_averages::ehlers_ecema::EhlersEcemaBuilder;
use my_project::indicators::moving_averages::ehlers_itrend::EhlersITrendBuilder;
use my_project::indicators::moving_averages::ehlers_kama::EhlersKamaBuilder;
use my_project::indicators::moving_averages::ehlers_pma::EhlersPmaBuilder;
use my_project::indicators::moving_averages::maaq::MaaqBuilder;
use my_project::indicators::moving_averages::mwdx::MwdxBuilder;
use my_project::indicators::moving_averages::buff_averages::BuffAveragesBuilder;

fn compute_ma_series_for(
    indicator: &str,
    req: &OptimizeRequest,
    period: usize,
    is_fast: bool,
    override_params: Option<&JsonValue>,
) -> Result<Vec<f64>> {
    let data_close: Vec<f64> = if let Some(c) = &req.close { c.clone() } else { req.series.clone().unwrap_or_else(|| gen_synthetic(100_000)) };
    let name = indicator.to_ascii_lowercase();
    // Pull per-indicator params
    let params = override_params.or_else(|| if is_fast { req.fast_params.as_ref() } else { req.slow_params.as_ref() });
    let values = match name.as_str() {
        "cwma" => { CwmaBuilder::default().period(period).apply_slice(&data_close)?.values }
        "edcf" => { EdcfBuilder::default().period(period).apply_slice(&data_close)?.values }
        "ehma" => { EhmaBuilder::default().period(period).apply_slice(&data_close)?.values }
        "maaq" => { MaaqBuilder::default().period(period).apply_slice(&data_close)?.values }
        "mwdx" => { MwdxBuilder::default().apply_slice(&data_close)?.values }
        "mama" => {
            // Use MAMA line as single-series MA (defaults)
            use my_project::indicators::moving_averages::mama::MamaBuilder;
            MamaBuilder::default().apply_slice(&data_close)?.mama_values
        }
        "ehlers_pma" => {
            // Use predict as primary series
            EhlersPmaBuilder::default().apply_slice(&data_close)?.predict
        }
        // Dispatcher/stream placeholders -> map to SMA for demo
        "ma" | "ma_stream" => { SmaBuilder::default().period(period).apply_slice(&data_close)?.values }
        "alma" => {
            let offset = req.offset; let sigma = req.sigma;
            AlmaBuilder::default().period(period).offset(offset).sigma(sigma).apply_slice(&data_close)?.values
        }
        "ema" => { EmaBuilder::default().period(period).apply_slice(&data_close)?.values }
        "sma" => { SmaBuilder::default().period(period).apply_slice(&data_close)?.values }
        "wma" => { WmaBuilder::default().period(period).apply_slice(&data_close)?.values }
        "zlema" => { ZlemaBuilder::default().period(period).apply_slice(&data_close)?.values }
        "dema" => { DemaBuilder::default().period(period).apply_slice(&data_close)?.values }
        "trima" => { TrimaBuilder::default().period(period).apply_slice(&data_close)?.values }
        "hma" => { HmaBuilder::default().period(period).apply_slice(&data_close)?.values }
        "smma" => { SmmaBuilder::default().period(period).apply_slice(&data_close)?.values }
        "dma" => {
            let mut b = DmaBuilder::default().hull_length(period).ema_length(period);
            if let Some(p) = params {
                if let Some(s) = p.get("hull_ma_type").and_then(|v| v.as_str()) { b = b.hull_ma_type(s.to_string()); }
            }
            b.apply_slice(&data_close)?.values
        }
        "swma" => { SwmaBuilder::default().period(period).apply_slice(&data_close)?.values }
        "sqwma" => { SqwmaBuilder::default().period(period).apply_slice(&data_close)?.values }
        "srwma" => { SrwmaBuilder::default().period(period).apply_slice(&data_close)?.values }
        "supersmoother" => { SuperSmootherBuilder::default().period(period).apply_slice(&data_close)?.values }
        "supersmoother_3_pole" => { SuperSmoother3PoleBuilder::default().period(period).apply_slice(&data_close)?.values }
        "sinwma" => { SinWmaBuilder::default().period(period).apply_slice(&data_close)?.values }
        "pwma" => { PwmaBuilder::default().period(period).apply_slice(&data_close)?.values }
        "vwma" => { let vols: Vec<f64> = if let Some(v) = &req.volume { v.clone() } else { vec![1.0; data_close.len()] }; VwmaBuilder::default().period(period).apply_slice(&data_close, &vols)?.values }
        "wilders" => { WildersBuilder::default().period(period).apply_slice(&data_close)?.values }
        "fwma" => { FwmaBuilder::default().period(period).apply_slice(&data_close)?.values }
        "linreg" => { LinRegBuilder::default().period(period).apply_slice(&data_close)?.values }
        "tema" => { TemaBuilder::default().period(period).apply_slice(&data_close)?.values }
        // Additional period-based MAs (defaults used for extra params when applicable)
        "jma" => {
            let mut b = JmaBuilder::default().period(period);
            if let Some(p) = params {
                if let Some(ph) = p.get("phase").and_then(|v| v.as_f64()) { b = b.phase(ph); }
                if let Some(pow) = p.get("power").and_then(|v| v.as_u64()) { b = b.power(pow as u32); }
            }
            b.apply_slice(&data_close)?.values
        }
        "kama" => { KamaBuilder::default().period(period).apply_slice(&data_close)?.values }
        "epma" => { EpmaBuilder::default().period(period).apply_slice(&data_close)?.values }
        "gaussian" => { GaussianBuilder::default().period(period).apply_slice(&data_close)?.values }
        "highpass" => { HighPassBuilder::default().period(period).apply_slice(&data_close)?.values }
        "highpass_2_pole" => { HighPass2Builder::default().period(period).apply_slice(&data_close)?.values }
        "hwma" => { HwmaBuilder::default().apply_slice(&data_close)?.values }
        "jsa" => { JsaBuilder::default().period(period).apply_slice(&data_close)?.values }
        "nama" => { NamaBuilder::default().period(period).apply_slice(&data_close)?.values }
        "nma" => { NmaBuilder::default().period(period).apply_slice(&data_close)?.values }
        "reflex" => { ReflexBuilder::default().period(period).apply_slice(&data_close)?.values }
        "sama" => {
            let mut b = SamaBuilder::default().length(period);
            if let Some(p) = params {
                if let Some(maj) = p.get("maj_length").and_then(|v| v.as_u64()) { b = b.maj_length(maj as usize); }
                if let Some(minl) = p.get("min_length").and_then(|v| v.as_u64()) { b = b.min_length(minl as usize); }
            }
            b.apply_slice(&data_close)?.values
        }
        "tilson" => { TilsonBuilder::default().period(period).apply_slice(&data_close)?.values }
        "tradjema" => {
            // Requires OHLC; use synthetic high/low
            let close = &data_close;
            let mut high=Vec::with_capacity(close.len()); let mut low=Vec::with_capacity(close.len());
            for (i,&c) in close.iter().enumerate(){ if c.is_finite(){ let amp=0.002*((i%100) as f64/100.0+0.5); high.push(c*(1.0+amp)); low.push(c*(1.0-amp)); } else { high.push(f64::NAN); low.push(f64::NAN);} }
            TradjemaBuilder::default().length(period).apply_slices(&high, &low, &close)?.values
        }
        "trendflex" => { TrendFlexBuilder::default().period(period).apply_slice(&data_close)?.values }
        "uma" => {
            let mut b = UmaBuilder::default().max_length(period);
            if let Some(p) = params {
                if let Some(a) = p.get("accelerator").and_then(|v| v.as_f64()) { b = b.accelerator(a); }
                if let Some(minl) = p.get("min_length").and_then(|v| v.as_u64()) { b = b.min_length(minl as usize); }
                if let Some(smooth) = p.get("smooth_length").and_then(|v| v.as_u64()) { b = b.smooth_length(smooth as usize); }
            }
            b.apply_slice(&data_close, None)?.values
        }
        "vpwma" => { VpwmaBuilder::default().period(period).apply_slice(&data_close)?.values }
        // Ehlers family
        "ehlers_ecema" => {
            let mut b = EhlersEcemaBuilder::default().length(period);
            if let Some(p) = params {
                if let Some(gl) = p.get("gain_limit").and_then(|v| v.as_u64()) { b = b.gain_limit(gl as usize); }
            }
            b.apply_slice(&data_close)?.values
        }
        "ehlers_itrend" => {
            let mut b = EhlersITrendBuilder::default().max_dc_period(period);
            if let Some(p) = params {
                if let Some(warm) = p.get("warmup_bars").and_then(|v| v.as_u64()) { b = b.warmup_bars(warm as usize); }
            }
            b.apply_slice(&data_close)?.values
        }
        "ehlers_kama" => { EhlersKamaBuilder::default().period(period).apply_slice(&data_close)?.values }
        // FRAMA: requires OHLC; use provided or derive synthetic high/low from close for demo
        "frama" => {
            let close = &data_close;
            let (high, low) = if let (Some(h), Some(l)) = (&req.high, &req.low) { (h.clone(), l.clone()) } else {
                // derive simple synthetic high/low around close
                let mut h=Vec::with_capacity(close.len()); let mut l=Vec::with_capacity(close.len());
                for (i,&c) in close.iter().enumerate(){ if c.is_finite(){ let amp=0.002*((i%100) as f64/100.0+0.5); h.push(c*(1.0+amp)); l.push(c*(1.0-amp)); } else { h.push(f64::NAN); l.push(f64::NAN);} }
                (h,l)
            };
            let sc = params.and_then(|p| p.get("sc")).and_then(|v| v.as_u64()).unwrap_or(300) as usize;
            let fc = params.and_then(|p| p.get("fc")).and_then(|v| v.as_u64()).unwrap_or(1) as usize;
            FramaBuilder::default().window(period).sc(sc).fc(fc).apply_slices(&high, &low, &close)?.values
        }
        // VWAP: requires timestamps, volumes, prices
        "vwap" => {
            let prices = &data_close;
            let timestamps: Vec<i64> = if let Some(ts) = &req.timestamps { ts.clone() } else {
                (0..prices.len()).map(|i| i as i64).collect()
            };
            let volumes: Vec<f64> = if let Some(v) = &req.volume { v.clone() } else { vec![1.0; prices.len()] };
            let anchor = req.anchor.clone().unwrap_or_else(|| "1d".to_string());
            VwapBuilder::default().anchor(anchor).apply_slice(&timestamps, &volumes, &prices)?.values
        }
        // Volume-adjusted MA (requires volume): use provided volume or ones for demo
        "volume_adjusted_ma" => {
            let prices = &data_close;
            let volume: Vec<f64> = if let Some(v) = &req.volume { v.clone() } else { vec![1.0; prices.len()] };
            VolumeAdjustedMaBuilder::default().length(period).apply_slices(&prices, &volume)?.values
        }
        // Volatility-adjusted MA: map grid period -> base_period and a derived vol_period
        "vama" | "volatility_adjusted_ma" => {
            let base_p = period.max(2);
            let vol_p = (period / 2).max(2);
            VamaBuilder::default().base_period(base_p).vol_period(vol_p).apply_slice(&data_close)?.values
        }
        _ => return Err(anyhow::anyhow!(format!("unsupported indicator: {}", indicator)))
    };
    Ok(values)
}

pub fn run_cpu(req: OptimizeRequest) -> Result<OptimizeResponse> {
    let data = match (&req.series, req.synthetic_len) {
        (Some(s), _) => s.clone(),
        (None, Some(n)) => gen_synthetic(n),
        _ => gen_synthetic(100_000),
    };
    let fast_periods = expand_range(req.fast_period);
    let slow_periods = expand_range(req.slow_period);
    let fast_type = req.fast_type.clone().unwrap_or_else(|| "alma".to_string());
    let slow_type = req.slow_type.clone().unwrap_or_else(|| "alma".to_string());
    let rows = fast_periods.len();
    let cols = slow_periods.len();

    // ──────────────────────────────────────────────────────────────
    // Multi-parameter sweep (CPU): ALMA offset/sigma
    // Only triggers when both sides select ALMA.
    // ──────────────────────────────────────────────────────────────
    if fast_type.to_ascii_lowercase() == "alma" && slow_type.to_ascii_lowercase() == "alma" {
        fn values_or_range(p: &Option<JsonValue>, key: &str, default_val: f64) -> Vec<f64> {
            if let Some(j) = p {
                if let Some(v) = j.get(key) {
                    if let Some(arr) = v.as_array() {
                        let mut out = Vec::new();
                        for x in arr { if let Some(f) = x.as_f64() { out.push(f) } }
                        if !out.is_empty() { return out; }
                    }
                    if v.is_object() {
                        let s = v.get("start").and_then(|x| x.as_f64()).unwrap_or(default_val);
                        let e = v.get("end").and_then(|x| x.as_f64()).unwrap_or(s);
                        let st = v.get("step").and_then(|x| x.as_f64()).unwrap_or(0.0);
                        if st.abs() > 0.0 && e >= s {
                            let mut out = Vec::new();
                            let mut x = s;
                            while x <= e + 1e-12 { out.push(x); x += st; }
                            if !out.is_empty() { return out; }
                        }
                    }
                    if let Some(f) = v.as_f64() { return vec![f]; }
                }
            }
            vec![default_val]
        }

        let f_offs = values_or_range(&req.fast_params, "offset", req.offset);
        let f_sigs = values_or_range(&req.fast_params, "sigma", req.sigma);
        let s_offs = values_or_range(&req.slow_params, "offset", req.offset);
        let s_sigs = values_or_range(&req.slow_params, "sigma", req.sigma);

        let f_ext = f_offs.len() * f_sigs.len();
        let s_ext = s_offs.len() * s_sigs.len();
        let layers = f_ext * s_ext;
        let m = req.metrics.max(5).min(5);
        let mut values = vec![0f32; layers * rows * cols * m];

        // Precompute fast banks per (offset,sigma)
        let mut fast_banks: Vec<Vec<Vec<f64>>> = Vec::with_capacity(f_ext);
        for &off in &f_offs {
            for &sig in &f_sigs {
                let ov = JsonValue::Object(serde_json::map::Map::from_iter(vec![
                    ("offset".to_string(), JsonValue::from(off)),
                    ("sigma".to_string(), JsonValue::from(sig)),
                ]));
                let bank: Vec<Vec<f64>> = fast_periods.par_iter().map(|&p|
                    compute_ma_series_for(&fast_type, &req, p, true, Some(&ov)).unwrap()
                ).collect();
                fast_banks.push(bank);
            }
        }
        // Precompute slow banks per (offset,sigma)
        let mut slow_banks: Vec<Vec<Vec<f64>>> = Vec::with_capacity(s_ext);
        for &off in &s_offs {
            for &sig in &s_sigs {
                let ov = JsonValue::Object(serde_json::map::Map::from_iter(vec![
                    ("offset".to_string(), JsonValue::from(off)),
                    ("sigma".to_string(), JsonValue::from(sig)),
                ]));
                let bank: Vec<Vec<f64>> = slow_periods.par_iter().map(|&p|
                    compute_ma_series_for(&slow_type, &req, p, false, Some(&ov)).unwrap()
                ).collect();
                slow_banks.push(bank);
            }
        }

        for (fi, fbank) in fast_banks.iter().enumerate() {
            for (si, sbank) in slow_banks.iter().enumerate() {
                let layer = fi * s_ext + si;
                for i in 0..rows {
                    let f_ma = &fbank[i];
                    for j in 0..cols {
                        let s_ma = &sbank[j];
                        let metrics = backtest_pair(&data, f_ma, s_ma, fast_periods[i], slow_periods[j], req.commission as f64);
                        let base = (((layer * rows + i) * cols + j) * m) as usize;
                        values[base..base+m].copy_from_slice(&metrics[..m]);
                    }
                }
            }
        }

        let axes = vec![
            AxisMeta { name: "fast_period".to_string(), values: fast_periods.iter().map(|&x| x as f64).collect() },
            AxisMeta { name: "slow_period".to_string(), values: slow_periods.iter().map(|&x| x as f64).collect() },
            AxisMeta { name: "fast.offset".to_string(), values: f_offs.clone() },
            AxisMeta { name: "fast.sigma".to_string(), values: f_sigs.clone() },
            AxisMeta { name: "slow.offset".to_string(), values: s_offs.clone() },
            AxisMeta { name: "slow.sigma".to_string(), values: s_sigs.clone() },
        ];

        let meta = OptimizeResponseMeta {
            fast_periods,
            slow_periods,
            metrics: vec!["total_return", "trades", "max_dd", "mean_ret", "std_ret"],
            rows,
            cols,
            axes,
        };
        return Ok(OptimizeResponse { meta, values, layers });
    }

    // Special-case: buff_averages requires both fast and slow periods together.
    if fast_type.to_ascii_lowercase() == "buff_averages" && slow_type.to_ascii_lowercase() == "buff_averages" {
        let prices = data;
        let volume: Vec<f64> = if let Some(v) = &req.volume { v.clone() } else { vec![1.0; prices.len()] };
        let m = req.metrics.max(5).min(5);
        let mut values = vec![0f32; rows * cols * m];
        // Brute-force over grid, invoking once per pair
        for (i_row, &fp) in fast_periods.iter().enumerate() {
            for (j_col, &sp) in slow_periods.iter().enumerate() {
                let out = BuffAveragesBuilder::default()
                    .fast_period(fp)
                    .slow_period(sp)
                    .apply_slices(&prices, &volume)?;
                let metrics = backtest_pair(&prices, &out.fast_buff, &out.slow_buff, fp, sp, req.commission as f64);
                let base = (i_row * cols + j_col) * m;
                values[base..base+m].copy_from_slice(&metrics[..m]);
            }
        }
        let axes = vec![
            AxisMeta { name: "fast_period".to_string(), values: fast_periods.iter().map(|&x| x as f64).collect() },
            AxisMeta { name: "slow_period".to_string(), values: slow_periods.iter().map(|&x| x as f64).collect() },
        ];
        let meta = OptimizeResponseMeta {
            fast_periods, slow_periods,
            metrics: vec!["total_return","trades","max_dd","mean_ret","std_ret"],
            rows, cols,
            axes,
        };
        return Ok(OptimizeResponse { meta, values, layers: 1 });
    }

    // Precompute fast and slow MA banks (time-major arrays per param)
    let fast_bank: Vec<Vec<f64>> = fast_periods.par_iter().map(|&p|
        compute_ma_series_for(&fast_type, &req, p, true, None).unwrap()
    ).collect();
    let slow_bank: Vec<Vec<f64>> = slow_periods.par_iter().map(|&p|
        compute_ma_series_for(&slow_type, &req, p, false, None).unwrap()
    ).collect();

    // Compute metrics per pair
    let m = req.metrics.max(5).min(5);
    let mut values = vec![0f32; rows * cols * m];
    values.par_chunks_mut(cols * m).enumerate().for_each(|(i_row, row_chunk)| {
        let f_ma = &fast_bank[i_row];
        for j in 0..cols {
            let s_ma = &slow_bank[j];
            let metrics = backtest_pair(&data, f_ma, s_ma, fast_periods[i_row], slow_periods[j], req.commission as f64);
            let base = j * m;
            row_chunk[base..base + m].copy_from_slice(&metrics[..m]);
        }
    });

    let axes = vec![
        AxisMeta { name: "fast_period".to_string(), values: fast_periods.iter().map(|&x| x as f64).collect() },
        AxisMeta { name: "slow_period".to_string(), values: slow_periods.iter().map(|&x| x as f64).collect() },
    ];
    let meta = OptimizeResponseMeta {
        fast_periods,
        slow_periods,
        metrics: vec!["total_return", "trades", "max_dd", "mean_ret", "std_ret"],
        rows,
        cols,
        axes,
    };
    Ok(OptimizeResponse { meta, values, layers: 1 })
}
