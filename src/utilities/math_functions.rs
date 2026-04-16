#![allow(clippy::many_single_char_names)]

use std::f64::consts::LN_2;

#[inline(always)]
pub fn atan_fast(z: f64) -> f64 {
    const C0: f64 = 0.2447;
    const C1: f64 = 0.0663;
    const PIO4: f64 = std::f64::consts::FRAC_PI_4;
    const PIO2: f64 = std::f64::consts::FRAC_PI_2;

    let a = z.abs();
    if a <= 1.0 {
        let t = C1.mul_add(a, C0);
        PIO4.mul_add(z, z.mul_add(a - 1.0, t))
    } else {
        let inv = 1.0 / z;
        let t = C1.mul_add(inv.abs(), C0);
        let base = PIO4.mul_add(inv, inv.mul_add(inv.abs() - 1.0, t));
        if z.is_sign_positive() {
            PIO2 - base
        } else {
            -PIO2 - base
        }
    }
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
pub fn exp_approx_f64(p: f64) -> f64 {
    const INV_LN2: f64 = std::f64::consts::LOG2_E;
    pow2_approx_f64(INV_LN2 * p)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
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
}
