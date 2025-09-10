use my_project::indicators::moving_averages::ehlers_itrend::{
    ehlers_itrend, EhlersITrendInput, EhlersITrendParams,
};

fn main() {
    // Case from failing proptest
    let data: Vec<f64> = vec![
        -474023.92896634823, 468163.34106897126, -201837.88805241624, -237506.6506212652,
        11632.529327397013, -694656.6131738159, -814848.1357115462, -727619.2307005139,
        59160.06173100908, 399849.37284927256, -18744.727476766664, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let warmup = 15usize;
    let max_dc = 44usize;
    let a = -26.304974549692695f64;
    let b = -740.1083765353759f64;

    let params = EhlersITrendParams { warmup_bars: Some(warmup), max_dc_period: Some(max_dc) };
    let input = EhlersITrendInput::from_slice(&data, params.clone());
    let base = ehlers_itrend(&input).unwrap().values;

    let transformed: Vec<f64> = data.iter().map(|x| a * x + b).collect();
    let t_vals = ehlers_itrend(&EhlersITrendInput::from_slice(&transformed, params)).unwrap().values;

    let i = warmup; // index under test
    let expected = a * base[i] + b;
    println!(
        "i={}, base={}, t_val={}, expected={}, diff={}",
        i, base[i], t_vals[i], expected, (t_vals[i] - expected).abs()
    );

    // Inspect internal 'it' accumulation at index i using a local replica
    fn inspect_it(data: &[f64], warmup_bars: usize, max_dc: usize, idx: usize) -> (f64, f64, f64, f64, f64, i32) {
        let mut fir_buf = [0.0; 7];
        let mut det_buf = [0.0; 7];
        let mut i1_buf = [0.0; 7];
        let mut q1_buf = [0.0; 7];
        let (mut prev_i2, mut prev_q2) = (0.0, 0.0);
        let (mut prev_re, mut prev_im) = (0.0, 0.0);
        let (mut prev_mesa, mut prev_smooth) = (0.0, 0.0);
        let mut sum_ring = vec![0.0; max_dc];
        let mut sum_idx = 0usize;
        let (mut prev_it1, mut prev_it2, mut prev_it3) = (0.0, 0.0, 0.0);
        let mut ring_ptr = 0usize;

        let mut y_val = f64::NAN;
        let mut it_curr = 0.0;
        let _it1 = 0.0; let _it2 = 0.0; let _it3 = 0.0;

        for i in 0..=idx {
            let x0 = data[i];
            let x1 = if i >= 1 { data[i - 1] } else { 0.0 };
            let x2 = if i >= 2 { data[i - 2] } else { 0.0 };
            let x3 = if i >= 3 { data[i - 3] } else { 0.0 };

            let fir_val = (4.0 * x0 + 3.0 * x1 + 2.0 * x2 + x3) / 10.0;
            fir_buf[ring_ptr] = fir_val;
            let get_ring = |buf: &[f64; 7], center: usize, offset: usize| -> f64 { buf[(7 + center - offset) % 7] };
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

            let i2_cur = 0.2 * (i1_val - j_q_val) + 0.8 * prev_i2;
            let q2_cur = 0.2 * (q1_val + j_i_val) + 0.8 * prev_q2;

            let re_val = i2_cur * prev_i2 + q2_cur * prev_q2;
            let im_val = i2_cur * prev_q2 - q2_cur * prev_i2;
            prev_i2 = i2_cur;
            prev_q2 = q2_cur;

            let re_smooth = 0.2 * re_val + 0.8 * prev_re;
            let im_smooth = 0.2 * im_val + 0.8 * prev_im;
            prev_re = re_smooth;
            prev_im = im_smooth;

            let mut new_mesa = if re_smooth != 0.0 && im_smooth != 0.0 {
                2.0 * std::f64::consts::PI / (im_smooth / re_smooth).atan()
            } else { 0.0 };
            let up_lim = 1.5 * prev_mesa;
            if new_mesa > up_lim { new_mesa = up_lim; }
            let low_lim = 0.67 * prev_mesa;
            if new_mesa < low_lim { new_mesa = low_lim; }
            new_mesa = new_mesa.clamp(6.0, 50.0);
            let final_mesa = 0.2 * new_mesa + 0.8 * prev_mesa;
            prev_mesa = final_mesa;
            let sp_val = 0.33 * final_mesa + 0.67 * prev_smooth;
            prev_smooth = sp_val;

            let mut dcp = (sp_val + 0.5).floor() as i32;
            if dcp < 1 { dcp = 1; }
            if dcp as usize > max_dc { dcp = max_dc as i32; }

            sum_ring[sum_idx] = x0;
            sum_idx = (sum_idx + 1) % max_dc;
            let mut sum_src = 0.0;
            let mut idx2 = sum_idx;
            for _ in 0..dcp {
                idx2 = if idx2 == 0 { max_dc - 1 } else { idx2 - 1 };
                sum_src += sum_ring[idx2];
            }
            it_curr = sum_src / dcp as f64;

            // at idx, compute y using prev_it*
            if i == idx {
                y_val = if i < warmup_bars { x0 } else { (4.0 * it_curr + 3.0 * prev_it1 + 2.0 * prev_it2 + prev_it3) / 10.0 };
            }

            prev_it3 = prev_it2; prev_it2 = prev_it1; prev_it1 = it_curr;
            ring_ptr = (ring_ptr + 1) % 7;
        }

        let _last_dcp = {
            let _new_mesa = 0.0; // not used here; compute dcp same as last iter
            0 // placeholder
        };
        (y_val, it_curr, prev_it1, prev_it2, prev_it3, 0)
    }

    let (y_b, it_b, p1_b, p2_b, p3_b, _db) = inspect_it(&data, warmup, max_dc, i);
    let (y_t, it_t, p1_t, p2_t, p3_t, _dt) = inspect_it(&data.iter().map(|x| a * x + b).collect::<Vec<_>>(), warmup, max_dc, i);
    println!("DETAIL base: y={}, it={}, p1={}, p2={}, p3={}", y_b, it_b, p1_b, p2_b, p3_b);
    println!("DETAIL tran: y={}, it={}, p1={}, p2={}, p3={}", y_t, it_t, p1_t, p2_t, p3_t);
}
