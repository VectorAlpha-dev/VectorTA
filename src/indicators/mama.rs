use crate::utilities::math_functions::atan64;
use std::error::Error;
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct MamaParams {
    pub fast_limit: f64,
    pub slow_limit: f64,
}

impl Default for MamaParams {
    fn default() -> Self {
        Self {
            fast_limit: 0.5,
            slow_limit: 0.05,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MamaInput<'a> {
    pub data: &'a [f64],
    pub params: MamaParams,
}

impl<'a> MamaInput<'a> {
    #[inline]
    pub fn new(data: &'a [f64], params: MamaParams) -> Self {
        Self { data, params }
    }

    #[inline]
    pub fn with_default_params(data: &'a [f64]) -> Self {
        Self {
            data,
            params: MamaParams::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MamaOutput {
    pub mama_values: Vec<f64>,
    pub fama_values: Vec<f64>,
}

#[inline(always)]
fn hilbert(x0: f64, x2: f64, x4: f64, x6: f64) -> f64 {
    0.0962 * x0 + 0.5769 * x2 - 0.5769 * x4 - 0.0962 * x6
}

#[inline]
pub fn calculate_mama(input: &MamaInput) -> Result<MamaOutput, Box<dyn Error>> {
    let src = input.data;
    let len = src.len();
    if len < 10 {
        return Err("Not enough data".into());
    }

    let fast_limit = input.params.fast_limit;
    let slow_limit = input.params.slow_limit;

    let mut mama_values = vec![0.0; len];
    let mut fama_values = vec![0.0; len];

    let mut smooth_buf = [0.0; 7];
    let mut detrender_buf = [0.0; 7];
    let mut i1_buf = [0.0; 7];
    let mut q1_buf = [0.0; 7];

    for b in &mut smooth_buf {
        *b = src[0];
    }
    for b in &mut detrender_buf {
        *b = src[0];
    }
    for b in &mut i1_buf {
        *b = src[0];
    }
    for b in &mut q1_buf {
        *b = src[0];
    }

    let mut prev_mesa_period = 0.0;
    let mut prev_mama = src[0];
    let mut prev_fama = src[0];
    let mut prev_i2_sm = 0.0;
    let mut prev_q2_sm = 0.0;
    let mut prev_re = 0.0;
    let mut prev_im = 0.0;
    let mut prev_phase = 0.0;

    for i in 0..len {
        let src_i = src[i];
        let s1 = if i >= 1 { src[i - 1] } else { src_i };
        let s2 = if i >= 2 { src[i - 2] } else { src_i };
        let s3 = if i >= 3 { src[i - 3] } else { src_i };

        let smooth_val = (4.0 * src_i + 3.0 * s1 + 2.0 * s2 + s3) / 10.0;

        let idx = i % 7;
        smooth_buf[idx] = smooth_val;

        let x0 = smooth_buf[idx];
        let x2 = smooth_buf[(idx + 7 - 2) % 7];
        let x4 = smooth_buf[(idx + 7 - 4) % 7];
        let x6 = smooth_buf[(idx + 7 - 6) % 7];

        let mesa_period_mult = 0.075 * prev_mesa_period + 0.54;
        let dt_val = hilbert(x0, x2, x4, x6) * mesa_period_mult;

        detrender_buf[idx] = dt_val;
        let d0 = detrender_buf[idx];
        let d2 = detrender_buf[(idx + 7 - 2) % 7];
        let d4 = detrender_buf[(idx + 7 - 4) % 7];
        let d6 = detrender_buf[(idx + 7 - 6) % 7];

        let i1_val = if i >= 3 {
            detrender_buf[(idx + 7 - 3) % 7]
        } else {
            d0
        };

        i1_buf[idx] = i1_val;

        let q1_val = hilbert(d0, d2, d4, d6) * mesa_period_mult;
        q1_buf[idx] = q1_val;

        let i1_0 = i1_buf[idx];
        let i1_2 = i1_buf[(idx + 7 - 2) % 7];
        let i1_4 = i1_buf[(idx + 7 - 4) % 7];
        let i1_6 = i1_buf[(idx + 7 - 6) % 7];
        let j_i = hilbert(i1_0, i1_2, i1_4, i1_6) * mesa_period_mult;

        let q1_0 = q1_buf[idx];
        let q1_2 = q1_buf[(idx + 7 - 2) % 7];
        let q1_4 = q1_buf[(idx + 7 - 4) % 7];
        let q1_6 = q1_buf[(idx + 7 - 6) % 7];
        let j_q = hilbert(q1_0, q1_2, q1_4, q1_6) * mesa_period_mult;

        let i2 = i1_val - j_q;
        let q2 = q1_val + j_i;

        let i2_sm = 0.2 * i2 + 0.8 * prev_i2_sm;
        let q2_sm = 0.2 * q2 + 0.8 * prev_q2_sm;

        let re = 0.2 * (i2_sm * prev_i2_sm + q2_sm * prev_q2_sm) + 0.8 * prev_re;
        let im = 0.2 * (i2_sm * prev_q2_sm - q2_sm * prev_i2_sm) + 0.8 * prev_im;

        prev_i2_sm = i2_sm;
        prev_q2_sm = q2_sm;
        prev_re = re;
        prev_im = im;

        let mut cur_mesa = if re != 0.0 && im != 0.0 {
            2.0 * PI / atan64(im / re)
        } else {
            0.0
        };

        let pm = if i > 0 { prev_mesa_period } else { cur_mesa };
        if cur_mesa > 1.5 * pm {
            cur_mesa = 1.5 * pm;
        }
        if cur_mesa < 0.67 * pm {
            cur_mesa = 0.67 * pm;
        }
        if cur_mesa < 6.0 {
            cur_mesa = 6.0;
        } else if cur_mesa > 50.0 {
            cur_mesa = 50.0;
        }

        let cur_mesa_smooth = 0.2 * cur_mesa + 0.8 * pm;
        prev_mesa_period = cur_mesa_smooth;

        let mut cur_phase = 0.0;
        if i1_val != 0.0 {
            cur_phase = (180.0 / PI) * atan64(q1_val / i1_val)
        }

        let old_phase = prev_phase;
        let mut dp = old_phase - cur_phase;
        if dp < 1.0 {
            dp = 1.0;
        }
        prev_phase = cur_phase;

        let alpha = {
            let a = fast_limit / dp;
            if a < slow_limit {
                slow_limit
            } else {
                a
            }
        };

        let cur_mama = alpha * src_i + (1.0 - alpha) * prev_mama;
        let a2 = 0.5 * alpha;
        let cur_fama = a2 * cur_mama + (1.0 - a2) * prev_fama;

        prev_mama = cur_mama;
        prev_fama = cur_fama;

        mama_values[i] = cur_mama;
        fama_values[i] = cur_fama;
    }

    Ok(MamaOutput {
        mama_values,
        fama_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::data_loader::read_candles_from_csv;

    #[test]
    fn test_mama_accuracy_with_tolerance() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = MamaParams {
            fast_limit: 0.5,
            slow_limit: 0.05,
        };
        let input = MamaInput::new(close_prices, params);
        let result = calculate_mama(&input).expect("Failed to calculate MAMA");

        let mama_vals = &result.mama_values;
        let fama_vals = &result.fama_values;
        assert!(mama_vals.len() > 5 && fama_vals.len() > 5);

        let last_idx = mama_vals.len() - 5;
        let expected = [
            (59272.6126101837, 59904.82955384927),
            (59268.03197967452, 59888.90961449489),
            (59153.51598983726, 59705.06120833049),
            (59153.59019034539, 59691.27443288086),
            (59128.66068082812, 59677.20908907954),
        ];

        for (i, &(exp_mama, exp_fama)) in expected.iter().enumerate() {
            let got_mama = mama_vals[last_idx + i];
            let got_fama = fama_vals[last_idx + i];

            let mama_diff = (got_mama - exp_mama).abs() / exp_mama * 100.0;
            let fama_diff = (got_fama - exp_fama).abs() / exp_fama * 100.0;
            println!("{}: got_mama={}, got_fama={}", i, got_mama, got_fama);
            assert!(
                mama_diff < 0.01,
                "MAMA mismatch at {}: expected {}, got {}, diff {}%",
                i,
                exp_mama,
                got_mama,
                mama_diff
            );
            assert!(
                fama_diff < 0.01,
                "FAMA mismatch at {}: expected {}, got {}, diff {}%",
                i,
                exp_fama,
                got_fama,
                fama_diff
            );
        }
    }
}
