use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

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

#[inline]
pub fn fast_sin_f64(mut x: f64) -> f64 {
    const TWO_PI: f64 = std::f64::consts::PI * 2.0;
    x = x % TWO_PI;
    if x < -std::f64::consts::PI {
        x += TWO_PI;
    } else if x > std::f64::consts::PI {
        x -= TWO_PI;
    }
    const FOUROVERPI: f64 = 1.2732395447351627;      // 4 / π
    const FOUROVERPISQ: f64 = 0.40528473456935109;   // 4 / π²
    const Q: f64 = 0.77633023248007499;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();

    let mut y = FOUROVERPI * ax - FOUROVERPISQ * ax * ax;
    if sign < 0.0 {
        y = -y;
    }
    y * (Q + (1.0 - Q) * y.abs())
}
