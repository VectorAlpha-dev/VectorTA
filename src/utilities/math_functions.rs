#![allow(clippy::many_single_char_names)]

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, LN_2, PI};

#[inline(always)]
fn flip_sign_nonnan(x: f64, val: f64) -> f64 {
    if x.is_sign_negative() {
        -val
    } else {
        val
    }
}

#[inline(always)]
pub fn atan_raw64(x: f64) -> f64 {
    const N2: f64 = 0.273;
    (FRAC_PI_4 + N2 - N2 * x.abs()) * x
}

#[inline(always)]
pub fn atan64(x: f64) -> f64 {
    if x.abs() > 1.0 {
        debug_assert!(!x.is_nan());
        flip_sign_nonnan(x, FRAC_PI_2) - atan_raw64(1.0 / x)
    } else {
        atan_raw64(x)
    }
}

#[inline(always)]
pub fn fast_sin_f64(mut x: f64) -> f64 {
    const TWO_PI: f64 = 2.0 * PI;

    x %= TWO_PI;
    if x < -PI {
        x += TWO_PI;
    } else if x > PI {
        x -= TWO_PI;
    }

    const FOUROVERPI: f64 = 1.2732395447351627;
    const FOUROVERPISQ: f64 = 0.405_284_734_569_351_1;
    const Q: f64 = 0.776_330_232_480_075;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();

    let mut y = FOUROVERPI * ax - FOUROVERPISQ * ax * ax;
    if sign < 0.0 {
        y = -y;
    }
    y * (Q + (1.0 - Q) * y.abs())
}

#[inline(always)]
pub fn fast_cos_f64(mut x: f64) -> f64 {
    const TWO_PI: f64 = 2.0 * PI;

    x %= TWO_PI;
    if x < -PI {
        x += TWO_PI;
    } else if x > PI {
        x -= TWO_PI;
    }

    x += FRAC_PI_2;
    if x > PI {
        x -= TWO_PI;
    } else if x < -PI {
        x += TWO_PI;
    }

    const FOUROVERPI: f64 = 1.2732395447351627;
    const FOUROVERPISQ: f64 = 0.405_284_734_569_351_1;
    const Q: f64 = 0.776_330_232_480_075;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();

    let mut y = FOUROVERPI * ax - FOUROVERPISQ * ax * ax;
    if sign < 0.0 {
        y = -y;
    }
    y * (Q + (1.0 - Q) * y.abs())
}

#[inline(always)]
fn to_bits_f64(x: f64) -> u64 {
    x.to_bits()
}
#[inline(always)]
fn from_bits_f64(u: u64) -> f64 {
    f64::from_bits(u)
}

#[inline]
pub fn log2_approx_f64(x: f64) -> f64 {
    let mut y = to_bits_f64(x) as f64;
    y *= 2.220446049250313e-16;
    y - 1022.94269504
}

#[inline]
pub fn ln_approx_f64(x: f64) -> f64 {
    log2_approx_f64(x) * LN_2
}

#[inline]
pub fn pow2_approx_f64(p: f64) -> f64 {
    let clipp = if p < -1022.0 { -1022.0 } else { p };
    const POW2_OFFSET: f64 = 1022.942695;
    let v = ((1u64 << 52) as f64 * (clipp + POW2_OFFSET)) as u64;
    from_bits_f64(v)
}

#[inline]
pub fn pow_approx_f64(x: f64, p: f64) -> f64 {
    pow2_approx_f64(p * log2_approx_f64(x))
}

#[inline]
pub fn exp_approx_f64(p: f64) -> f64 {
    const INV_LN2: f64 = std::f64::consts::LOG2_E;
    pow2_approx_f64(INV_LN2 * p)
}

#[inline]
pub fn sigmoid_approx_f64(x: f64) -> f64 {
    1.0 / (1.0 + exp_approx_f64(-x))
}

#[inline]
pub fn lambertw_approx_f64(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }

    let mut w = if x < 1.0 {
        x
    } else {
        let g = ln_approx_f64(x).max(0.0);
        if g < 0.5 {
            0.5
        } else {
            g
        }
    };

    for _ in 0..2 {
        let ew = exp_approx_f64(w);
        let f = w * ew - x;
        let fp = ew * (w + 1.0);
        w -= f / fp;
    }
    w
}

#[inline]
pub fn lambertwexpx_approx_f64(v: f64) -> f64 {
    let mut y = 1.0_f64 + v.abs();
    for _ in 0..5 {
        let w = lambertw_approx_f64(y);
        y = w * exp_approx_f64(w);
    }
    y
}

#[inline]
pub fn ln_gamma_approx_f64(x: f64) -> f64 {
    -0.0810614667_f64 - x - ln_approx_f64(x) + (0.5_f64 + x) * ln_approx_f64(1.0_f64 + x)
}

#[inline]
pub fn digamma_approx_f64(x: f64) -> f64 {
    let onepx = 1.0 + x;
    -1.0 / x - 1.0 / (2.0 * onepx) + ln_approx_f64(onepx)
}

#[inline]
pub fn erfc_approx_f64(x: f64) -> f64 {
    const K: f64 = 3.3509633149424609;
    2.0 / (1.0 + pow2_approx_f64(K * x))
}

#[inline]
pub fn erf_approx_f64(x: f64) -> f64 {
    1.0 - erfc_approx_f64(x)
}

#[inline]
pub fn erf_inv_approx_f64(x: f64) -> f64 {
    const INVK: f64 = 0.30004578719350504;
    let ratio = (1.0 + x) / (1.0 - x);
    INVK * log2_approx_f64(ratio)
}

#[inline]
pub fn sinh_approx_f64(x: f64) -> f64 {
    0.5 * (exp_approx_f64(x) - exp_approx_f64(-x))
}

#[inline]
pub fn cosh_approx_f64(x: f64) -> f64 {
    0.5 * (exp_approx_f64(x) + exp_approx_f64(-x))
}

#[inline]
pub fn tanh_approx_f64(x: f64) -> f64 {
    -1.0 + 2.0 / (1.0 + exp_approx_f64(-2.0 * x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_fast_sin_cos() {
        let angles = [
            0.0,
            PI * 0.25,
            PI * 0.5,
            PI * 0.75,
            PI,
            -PI * 0.5,
            -PI,
            10.0,
            -10.0,
        ];
        for &ang in &angles {
            let fs = fast_sin_f64(ang);
            let fc = fast_cos_f64(ang);
            let rs = ang.sin();
            let rc = ang.cos();

            assert!(
                approx_eq(fs, rs, 0.05),
                "fast_sin_f64({ang}) => {fs} vs std => {rs}"
            );
            assert!(
                approx_eq(fc, rc, 0.05),
                "fast_cos_f64({ang}) => {fc} vs std => {rc}"
            );
        }
    }

    #[test]
    fn test_atan_approx() {
        let vals = [0.0, 0.5, 1.0, 2.0, -1.0, -10.0];
        for &v in &vals {
            let app = atan64(v);
            let real = v.atan();
            assert!(
                approx_eq(app, real, 0.1),
                "atan64({v}) => {app}, real => {real}"
            );
        }
    }

    #[test]
    fn test_log2_approx() {
        let vals = [0.125, 0.5, 1.0, 2.0, 8.0, 10.0];
        for &v in &vals {
            let app = log2_approx_f64(v);
            let real = v.log2();
            assert!(
                approx_eq(app, real, 0.15),
                "log2_approx_f64({v}) => {app}, real => {real}"
            );
        }
    }

    #[test]
    fn test_ln_approx() {
        let vals = [0.125, 0.5, 1.0, 2.0, 8.0, 10.0];
        for &v in &vals {
            let app = ln_approx_f64(v);
            let real = v.ln();
            assert!(
                approx_eq(app, real, 0.2),
                "ln_approx_f64({v}) => {app}, real => {real}"
            );
        }
    }

    #[test]
    fn test_exp_approx() {
        let vals = [-2.0, -1.0, 0.0, 1.0, 2.0, 5.0];
        for &v in &vals {
            let app = exp_approx_f64(v);
            let real = v.exp();
            let tol = 0.15 * real.abs().max(1.0);
            assert!(
                approx_eq(app, real, tol),
                "exp_approx_f64({v}) => {app}, real => {real}"
            );
        }
    }

    #[test]
    fn test_pow2_approx() {
        let vals = [-10.0, -1.0, 0.0, 1.0, 10.0, 15.5];
        for &v in &vals {
            let app = pow2_approx_f64(v);
            let real = (2.0_f64).powf(v);
            let tol = 0.15 * real.abs().max(1.0);
            assert!(
                approx_eq(app, real, tol),
                "pow2_approx_f64({v}) => {app}, real => {real}"
            );
        }
    }

    #[test]
    fn test_pow_approx() {
        let bases = [0.5, 1.0, 2.0, 10.0];
        let exps = [-2.0, -1.0, 0.0, 1.0, 2.0];
        for &b in &bases {
            for &p in &exps {
                let app = pow_approx_f64(b, p);
                let real = b.powf(p);
                let tol = 0.20 * real.abs().max(1.0);
                assert!(
                    approx_eq(app, real, tol),
                    "pow_approx_f64({b}^{p}) => {app}, real => {real}"
                );
            }
        }
    }

    #[test]
    fn test_sigmoid_approx() {
        let vals = [-4.0, -1.0, 0.0, 1.0, 4.0];
        for &v in &vals {
            let app = sigmoid_approx_f64(v);
            let real = 1.0 / (1.0 + (-v).exp());
            assert!(
                approx_eq(app, real, 0.02),
                "sigmoid_approx_f64({v}) => {app}, real => {real}"
            );
        }
    }

    #[test]
    fn test_erf_inv_approx() {
        let vals = [-0.9, -0.5, 0.0, 0.5, 0.9];
        for &v in &vals {
            let y_approx = erf_inv_approx_f64(v);
            let check = erf_approx_f64(y_approx);
            assert!(
                approx_eq(check, v, 0.2),
                "erf_inv_approx_f64({v}) => {y_approx}, but erf_approx_f64 => {check}"
            );
        }
    }

    #[test]
    fn test_hyperbolic_approx() {
        let vals = [-2.0, -1.0, 0.0, 1.0, 2.0];
        for &v in &vals {
            let sh = sinh_approx_f64(v);
            let ch = cosh_approx_f64(v);
            let th = tanh_approx_f64(v);
            let tol_s = 0.15 * v.sinh().abs().max(1.0);
            let tol_c = 0.15 * v.cosh().abs().max(1.0);
            assert!(
                approx_eq(sh, v.sinh(), tol_s),
                "sinh_approx_f64({v}) => {sh}, real => {}",
                v.sinh()
            );
            assert!(
                approx_eq(ch, v.cosh(), tol_c),
                "cosh_approx_f64({v}) => {ch}, real => {}",
                v.cosh()
            );
            assert!(
                approx_eq(th, v.tanh(), 0.15),
                "tanh_approx_f64({v}) => {th}, real => {}",
                v.tanh()
            );
        }
    }

    #[test]
    fn test_lambertw_approx() {
        let xvals = [1.0_f64, std::f64::consts::E];
        let real = [0.5671432904097838, 1.0];
        for (i, &x) in xvals.iter().enumerate() {
            let app = lambertw_approx_f64(x);
            assert!(
                approx_eq(app, real[i], 0.2),
                "lambertw_approx_f64({x}) => {app}, real => {}",
                real[i]
            );
        }
    }

    #[test]
    fn test_lambertwexpx_approx() {
        let vals = [1.0, 2.0, 3.0];
        for &v in &vals {
            let y = lambertwexpx_approx_f64(v);
            let wtest = lambertw_approx_f64(y);
            let check = wtest * exp_approx_f64(wtest);
            assert!(
                approx_eq(check, y, 0.3 * y.max(1.0)),
                "lambertwexpx_approx_f64({v}) => {y}, but checking => {check}"
            );
        }
    }
}
