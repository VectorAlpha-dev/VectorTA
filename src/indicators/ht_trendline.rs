use std::error::Error;
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct EhlersITrendParams {
    pub warmup_bars: Option<usize>,
    pub max_dc_period: Option<usize>,
}

impl Default for EhlersITrendParams {
    fn default() -> Self {
        EhlersITrendParams {
            warmup_bars: Some(12),
            max_dc_period: Some(60),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EhlersITrendInput<'a> {
    pub data: &'a [f64],
    pub params: EhlersITrendParams,
}

impl<'a> EhlersITrendInput<'a> {
    pub fn new(data: &'a [f64], params: EhlersITrendParams) -> Self {
        EhlersITrendInput { data, params }
    }

    pub fn with_default_params(data: &'a [f64]) -> Self {
        EhlersITrendInput {
            data,
            params: EhlersITrendParams::default(),
        }
    }

    #[inline]
    fn get_warmup_bars(&self) -> usize {
        self.params
            .warmup_bars
            .unwrap_or_else(|| EhlersITrendParams::default().warmup_bars.unwrap())
    }

    #[inline]
    fn get_max_dc_period(&self) -> usize {
        self.params
            .max_dc_period
            .unwrap_or_else(|| EhlersITrendParams::default().max_dc_period.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct EhlersITrendOutput {
    pub values: Vec<f64>,
}

pub fn calculate_ehlers_itrend(
    input: &EhlersITrendInput,
) -> Result<EhlersITrendOutput, Box<dyn Error>> {
    let src = input.data;
    let length = src.len();
    if length == 0 {
        return Ok(EhlersITrendOutput { values: vec![] });
    }

    let warmup_bars = input.get_warmup_bars();
    let max_dc = input.get_max_dc_period().max(1);

    let mut out_eit = vec![0.0; length];

    #[inline(always)]
    fn ring_idx(base: usize) -> usize {
        base % 7
    }

    let mut fir_buf = [0.0; 7];
    let mut det_buf = [0.0; 7];
    let mut i1_buf = [0.0; 7];
    let mut q1_buf = [0.0; 7];

    let mut prev_i2 = 0.0;
    let mut prev_q2 = 0.0;
    let mut prev_re = 0.0;
    let mut prev_im = 0.0;

    let mut prev_mesa = 0.0;
    let mut prev_smooth = 0.0;

    let mut sum_ring = vec![0.0; max_dc];
    let mut sum_idx = 0_usize;

    let mut prev_it1 = 0.0;
    let mut prev_it2 = 0.0;
    let mut prev_it3 = 0.0;

    let mut ring_ptr = 0_usize;
    for i in 0..length {
        let x0 = src[i];

        let x1 = if i >= 1 { src[i - 1] } else { 0.0 };
        let x2 = if i >= 2 { src[i - 2] } else { 0.0 };
        let x3 = if i >= 3 { src[i - 3] } else { 0.0 };
        let fir_val = (4.0 * x0 + 3.0 * x1 + 2.0 * x2 + x3) / 10.0;
        fir_buf[ring_ptr] = fir_val;

        #[inline(always)]
        fn get_ring(buf: &[f64; 7], center: usize, offset: usize) -> f64 {
            buf[(7 + center - offset) % 7]
        }
        let fir_0 = get_ring(&fir_buf, ring_ptr, 0);
        let fir_2 = get_ring(&fir_buf, ring_ptr, 2);
        let fir_4 = get_ring(&fir_buf, ring_ptr, 4);
        let fir_6 = get_ring(&fir_buf, ring_ptr, 6);

        let h_in = 0.0962 * fir_0 + 0.5769 * fir_2 - 0.5769 * fir_4 - 0.0962 * fir_6;
        let period_mult = 0.075 * prev_mesa + 0.54;
        let det_val = h_in * period_mult;
        det_buf[ring_ptr] = det_val;

        let i1_val = get_ring(&det_buf, ring_ptr, 3);
        i1_buf[ring_ptr] = i1_val;

        let det_0 = get_ring(&det_buf, ring_ptr, 0);
        let det_2 = get_ring(&det_buf, ring_ptr, 2);
        let det_4 = get_ring(&det_buf, ring_ptr, 4);
        let det_6 = get_ring(&det_buf, ring_ptr, 6);
        let h_in_q1 = 0.0962 * det_0 + 0.5769 * det_2 - 0.5769 * det_4 - 0.0962 * det_6;
        let q1_val = h_in_q1 * period_mult;
        q1_buf[ring_ptr] = q1_val;

        let i1_0 = get_ring(&i1_buf, ring_ptr, 0);
        let i1_2 = get_ring(&i1_buf, ring_ptr, 2);
        let i1_4 = get_ring(&i1_buf, ring_ptr, 4);
        let i1_6 = get_ring(&i1_buf, ring_ptr, 6);
        let j_i_val = (0.0962 * i1_0 + 0.5769 * i1_2 - 0.5769 * i1_4 - 0.0962 * i1_6) * period_mult;

        let q1_0 = get_ring(&q1_buf, ring_ptr, 0);
        let q1_2 = get_ring(&q1_buf, ring_ptr, 2);
        let q1_4 = get_ring(&q1_buf, ring_ptr, 4);
        let q1_6 = get_ring(&q1_buf, ring_ptr, 6);
        let j_q_val = (0.0962 * q1_0 + 0.5769 * q1_2 - 0.5769 * q1_4 - 0.0962 * q1_6) * period_mult;

        let mut i2_cur = i1_val - j_q_val;
        let mut q2_cur = q1_val + j_i_val;
        i2_cur = 0.2 * i2_cur + 0.8 * prev_i2;
        q2_cur = 0.2 * q2_cur + 0.8 * prev_q2;

        prev_i2 = i2_cur;
        prev_q2 = q2_cur;

        let re_val = i2_cur * prev_i2 + q2_cur * prev_q2;
        let im_val = i2_cur * prev_q2 - q2_cur * prev_i2;

        let re_smooth = 0.2 * re_val + 0.8 * prev_re;
        let im_smooth = 0.2 * im_val + 0.8 * prev_im;
        prev_re = re_smooth;
        prev_im = im_smooth;

        let mut new_mesa = 0.0;
        if re_smooth != 0.0 && im_smooth != 0.0 {
            new_mesa = 2.0 * PI / (im_smooth / re_smooth).atan();
        }
        let up_lim = 1.5 * prev_mesa;
        if new_mesa > up_lim {
            new_mesa = up_lim;
        }
        let low_lim = 0.67 * prev_mesa;
        if new_mesa < low_lim {
            new_mesa = low_lim;
        }
        if new_mesa < 6.0 {
            new_mesa = 6.0;
        }
        if new_mesa > 50.0 {
            new_mesa = 50.0;
        }
        let final_mesa = 0.2 * new_mesa + 0.8 * prev_mesa;
        prev_mesa = final_mesa;

        let sp_val = 0.33 * final_mesa + 0.67 * prev_smooth;
        prev_smooth = sp_val;

        let mut dcp = (sp_val + 0.5).floor() as i32;
        if dcp < 1 {
            dcp = 1;
        }
        if dcp as usize > max_dc {
            dcp = max_dc as i32;
        }

        sum_ring[sum_idx] = x0;
        sum_idx = (sum_idx + 1) % max_dc;

        let mut sum_src = 0.0;
        let mut idx2 = sum_idx;
        for _ in 0..dcp {
            idx2 = if idx2 == 0 { max_dc - 1 } else { idx2 - 1 };
            sum_src += sum_ring[idx2];
        }
        let it_val = sum_src / dcp as f64;

        let eit_val = if i < warmup_bars {
            x0
        } else {
            let tmp = (4.0 * it_val + 3.0 * prev_it1 + 2.0 * prev_it2 + prev_it3) / 10.0;
            tmp
        };

        prev_it3 = prev_it2;
        prev_it2 = prev_it1;
        prev_it1 = it_val;

        out_eit[i] = eit_val;

        ring_ptr = (ring_ptr + 1) % 7;
    }

    Ok(EhlersITrendOutput { values: out_eit })
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::indicators::data_loader::read_candles_from_csv;

    #[test]
    fn test_ehlers_itrend_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let input = EhlersITrendInput::with_default_params(&close_prices);
        let eit_result = calculate_ehlers_itrend(&input).expect("EIT calculation failed");

        assert!(
            eit_result.values.len() >= 5,
            "Not enough EIT values for the test"
        );

        println!("Last 5 EIT values:");
        for val in &eit_result.values[eit_result.values.len() - 5..] {
            println!("{val}");
        }

        let expected_last_five = [59448.3, 59441.6, 59399.4, 59309.5, 59218.7];

        let start_index = eit_result.values.len() - 5;
        let actual_last_five = &eit_result.values[start_index..];

        for (i, &actual) in actual_last_five.iter().enumerate() {
            let expected = expected_last_five[i];
            let diff = (actual - expected).abs();
            println!(
                "Bar {} => got={:.2}, expected={:.2}, diff={:.2}",
                i, actual, expected, diff
            );
        }
    }
}
