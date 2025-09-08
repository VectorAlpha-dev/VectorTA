use my_project::other_indicators::prb::{prb, PrbInput, PrbParams};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use the exact same data as the Python test
    let data: Vec<f64> = vec![
        66982.0, 66984.0, 66981.0, 66975.0, 66970.0,
        66968.0, 66960.0, 66955.0, 66950.0, 66945.0,
        66940.0, 66935.0, 66930.0, 66925.0, 66920.0,
        66915.0, 66910.0, 66905.0, 66900.0, 66895.0,
        66890.0, 66885.0, 66880.0, 66875.0, 66870.0,
        66865.0, 66860.0, 66855.0, 66850.0, 66845.0,
        66840.0, 66835.0, 66830.0, 66825.0, 66820.0,
        66815.0, 66810.0, 66805.0, 66800.0, 66795.0,
        66790.0, 66785.0, 66780.0, 66775.0, 66770.0,
        66765.0, 66760.0, 66755.0, 66750.0, 66745.0,
        66740.0, 66735.0, 66730.0, 66725.0, 66720.0,
        66715.0, 66710.0, 66705.0, 66700.0, 66695.0,
        66690.0, 66685.0, 66680.0, 66675.0, 66670.0,
        66665.0, 66660.0, 66655.0, 66650.0, 66645.0,
        66640.0, 66635.0, 66630.0, 66625.0, 66620.0,
        66615.0, 66610.0, 66605.0, 66600.0, 66595.0,
        66590.0, 66585.0, 66580.0, 66575.0, 66570.0,
        66565.0, 66560.0, 66555.0, 66550.0, 66545.0,
        66540.0, 66535.0, 66530.0, 66525.0, 66520.0,
        66515.0, 66510.0, 66505.0, 66500.0, 66495.0,
        66490.0, 66485.0, 66480.0, 66475.0, 66470.0,
    ];
    
    // Create PRB input with default parameters (no smoothing)
    let params = PrbParams {
        smooth_data: Some(false),  // No smoothing as per user request
        smooth_period: None,
        regression_period: Some(100),  // Default period of 100
        polynomial_order: Some(2),
        regression_offset: Some(0),
        ndev: Some(2.0),
        equ_from: Some(0),
    };
    
    let input = PrbInput::from_slice(&data, params);
    let output = prb(&input)?;
    
    // Get the last 5 non-NaN values
    let non_nan_values: Vec<(usize, f64)> = output.values
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.is_nan())
        .map(|(i, v)| (i, *v))
        .collect();
    
    println!("PRB Reference Values for synthetic data (no smoothing, period=100):");
    println!("Last 5 main values:");
    let start = non_nan_values.len().saturating_sub(5);
    for (i, val) in &non_nan_values[start..] {
        println!("  Index {}: {:.8}", i, val);
    }
    
    // Also get upper and lower bands
    let non_nan_upper: Vec<(usize, f64)> = output.upper_band
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.is_nan())
        .map(|(i, v)| (i, *v))
        .collect();
    
    let non_nan_lower: Vec<(usize, f64)> = output.lower_band
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.is_nan())
        .map(|(i, v)| (i, *v))
        .collect();
    
    println!("\nLast 5 upper band values:");
    let start = non_nan_upper.len().saturating_sub(5);
    for (i, val) in &non_nan_upper[start..] {
        println!("  Index {}: {:.8}", i, val);
    }
    
    println!("\nLast 5 lower band values:");
    let start = non_nan_lower.len().saturating_sub(5);
    for (i, val) in &non_nan_lower[start..] {
        println!("  Index {}: {:.8}", i, val);
    }
    
    Ok(())
}