
use vector_ta::indicators::atr::{atr, AtrData, AtrInput, AtrParams};

#[test]
fn atr_basic_smoke() {
    
    let high = [2.0, 3.0, 4.0, 5.0, 6.0];
    let low  = [1.0, 1.5, 2.0, 3.5, 4.0];
    let close= [1.5, 2.5, 3.0, 4.0, 5.0];
    let input = AtrInput { data: AtrData::Slices { high: &high, low: &low, close: &close }, params: AtrParams { length: Some(3) } };
    let out = atr(&input).expect("atr computes");
    assert_eq!(out.values.len(), high.len());
    
    assert!(out.values[0].is_nan());
    assert!(out.values[1].is_nan());
    
    for v in &out.values[2..] { assert!(v.is_finite()); }
}

#[test]
fn atr_errors_mapped() {
    
    let empty: [f64;0] = [];
    let input = AtrInput { data: AtrData::Slices { high: &empty, low: &empty, close: &empty }, params: AtrParams { length: Some(14) } };
    let err = atr(&input).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("atr:") || msg.contains("Invalid length") || msg.contains("empty"));
}

