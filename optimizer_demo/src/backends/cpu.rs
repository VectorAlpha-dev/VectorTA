use rayon::prelude::*;
use anyhow::Result;
use crate::backends::types::{OptimizeRequest, OptimizeResponse, OptimizeResponseMeta, expand_range};
use my_project::indicators::moving_averages::alma::AlmaBuilder;
use my_project::indicators::moving_averages::ema::EmaBuilder;
use my_project::indicators::moving_averages::sma::SmaBuilder;
use my_project::indicators::moving_averages::wma::WmaBuilder;
use my_project::indicators::moving_averages::zlema::ZlemaBuilder;
use my_project::indicators::moving_averages::dema::DemaBuilder;
use my_project::indicators::moving_averages::trima::TrimaBuilder;
use my_project::indicators::moving_averages::hma::HmaBuilder;
use my_project::indicators::moving_averages::smma::SmmaBuilder;
use my_project::indicators::moving_averages::dma::DmaBuilder;
use my_project::indicators::moving_averages::swma::SwmaBuilder;
use my_project::indicators::moving_averages::sqwma::SqwmaBuilder;
use my_project::indicators::moving_averages::srwma::SrwmaBuilder;
use my_project::indicators::moving_averages::supersmoother::SuperSmootherBuilder;
use my_project::indicators::moving_averages::supersmoother_3_pole::SuperSmoother3PoleBuilder;
use my_project::indicators::moving_averages::sinwma::SinWmaBuilder;
use my_project::indicators::moving_averages::pwma::PwmaBuilder;
use my_project::indicators::moving_averages::vwma::VwmaBuilder;
use my_project::indicators::moving_averages::wilders::WildersBuilder;
use my_project::indicators::moving_averages::fwma::FwmaBuilder;
use my_project::indicators::moving_averages::linreg::LinRegBuilder;
use my_project::indicators::moving_averages::tema::TemaBuilder;
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

fn compute_ma_series_for(indicator: &str, req: &OptimizeRequest, period: usize, is_fast: bool) -> Result<Vec<f64>> {
    let data_close: Vec<f64> = if let Some(c) = &req.close { c.clone() } else { req.series.clone().unwrap_or_else(|| gen_synthetic(100_000)) };
    let name = indicator.to_ascii_lowercase();
    // Pull per-indicator params
    let params = if is_fast { req.fast_params.as_ref() } else { req.slow_params.as_ref() };
    let values = match name.as_str() {
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
        "dma" => { DmaBuilder::default().period(period).apply_slice(&data_close)?.values }
        "swma" => { SwmaBuilder::default().period(period).apply_slice(&data_close)?.values }
        "sqwma" => { SqwmaBuilder::default().period(period).apply_slice(&data_close)?.values }
        "srwma" => { SrwmaBuilder::default().period(period).apply_slice(&data_close)?.values }
        "supersmoother" => { SuperSmootherBuilder::default().period(period).apply_slice(&data_close)?.values }
        "supersmoother_3_pole" => { SuperSmoother3PoleBuilder::default().period(period).apply_slice(&data_close)?.values }
        "sinwma" => { SinWmaBuilder::default().period(period).apply_slice(&data_close)?.values }
        "pwma" => { PwmaBuilder::default().period(period).apply_slice(&data_close)?.values }
        "vwma" => { VwmaBuilder::default().period(period).apply_slice(&data_close)?.values }
        "wilders" => { WildersBuilder::default().period(period).apply_slice(&data_close)?.values }
        "fwma" => { FwmaBuilder::default().period(period).apply_slice(&data_close)?.values }
        "linreg" => { LinRegBuilder::default().period(period).apply_slice(&data_close)?.values }
        "tema" => { TemaBuilder::default().period(period).apply_slice(&data_close)?.values }
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
        _ => return Err(anyhow::anyhow!(format!("unsupported indicator: {}", indicator)))
    };
    Ok(values)
}

pub fn run_cpu(req: OptimizeRequest) -> Result<OptimizeResponse> {
    let data = match (req.series, req.synthetic_len) {
        (Some(s), _) => s,
        (None, Some(n)) => gen_synthetic(n),
        _ => gen_synthetic(100_000),
    };
    let fast_periods = expand_range(req.fast_period);
    let slow_periods = expand_range(req.slow_period);
    let fast_type = req.fast_type.clone().unwrap_or_else(|| "alma".to_string());
    let slow_type = req.slow_type.clone().unwrap_or_else(|| "alma".to_string());
    let rows = fast_periods.len();
    let cols = slow_periods.len();

    // Precompute fast and slow MA banks (time-major arrays per param)
    let fast_bank: Vec<Vec<f64>> = fast_periods.par_iter().map(|&p|
        compute_ma_series_for(&fast_type, &req, p, true).unwrap()
    ).collect();
    let slow_bank: Vec<Vec<f64>> = slow_periods.par_iter().map(|&p|
        compute_ma_series_for(&slow_type, &req, p, false).unwrap()
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

    let meta = OptimizeResponseMeta {
        fast_periods,
        slow_periods,
        metrics: vec!["total_return", "trades", "max_dd", "mean_ret", "std_ret"],
        rows,
        cols,
    };
    Ok(OptimizeResponse { meta, values })
}
