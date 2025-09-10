use std::env;

fn main() {
    println!("TA-LIB Integration Test");
    println!("========================\n");
    
    // Check if TALIB_PATH is set
    match env::var("TALIB_PATH") {
        Ok(path) => {
            println!("✓ TALIB_PATH is set to: {}", path);
            
            // Check if the path exists
            let lib_path = std::path::Path::new(&path);
            if lib_path.exists() {
                println!("✓ TA-LIB directory exists");
                
                // Check for include and lib directories
                let include_path = lib_path.join("include");
                let lib_dir_path = lib_path.join("lib");
                
                if include_path.exists() {
                    println!("✓ Include directory found: {}", include_path.display());
                } else {
                    println!("✗ Include directory not found at: {}", include_path.display());
                }
                
                if lib_dir_path.exists() {
                    println!("✓ Library directory found: {}", lib_dir_path.display());
                    
                    // List library files
                    if let Ok(entries) = std::fs::read_dir(&lib_dir_path) {
                        println!("\nLibrary files found:");
                        for entry in entries.flatten() {
                            if let Some(name) = entry.file_name().to_str() {
                                if name.ends_with(".lib") || name.ends_with(".dll") {
                                    println!("  - {}", name);
                                }
                            }
                        }
                    }
                } else {
                    println!("✗ Library directory not found at: {}", lib_dir_path.display());
                }
            } else {
                println!("✗ TA-LIB directory does not exist at: {}", path);
                println!("\nPlease check your installation and TALIB_PATH setting.");
            }
        }
        Err(_) => {
            println!("✗ TALIB_PATH environment variable is not set!");
            println!("\nTo set it temporarily in PowerShell:");
            println!("  $env:TALIB_PATH = \"C:\\Program Files\\TA-Lib\"");
            println!("\nTo set it permanently:");
            println!("  1. Open System Properties → Environment Variables");
            println!("  2. Add new system variable:");
            println!("     Name: TALIB_PATH");
            println!("     Value: C:\\Program Files\\TA-Lib (or your installation path)");
        }
    }
    
    // Try to use TA-LIB if available (when compiled with the feature)
    #[cfg(feature = "talib")]
    {
        println!("\n✓ Compiled with TA-LIB support");
        test_talib_function();
    }
    
    #[cfg(not(feature = "talib"))]
    {
        println!("\n✗ Not compiled with TA-LIB support");
        println!("  Rebuild with: cargo build --features talib");
    }
}

#[cfg(feature = "talib")]
fn test_talib_function() {
    // This would test actual TA-LIB functionality when bindings are available
    println!("Testing TA-LIB function calls...");
    
    // Example test data
    let test_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    println!("Test data: {:?}", test_data);
    
    // When TA-LIB is properly integrated, you would call functions here
    // For example:
    // let result = unsafe { TA_SMA(...) };
    
    println!("TA-LIB function test would run here once bindings are generated.");
}