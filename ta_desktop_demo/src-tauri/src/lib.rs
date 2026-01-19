pub mod commands;
pub mod ma_registry;
pub mod state;

use commands::*;
use state::AppState;

pub fn run_tauri() {
    tauri::Builder::default()
        .manage(AppState::default())
        .setup(|app| {
            let window_cfg = app
                .config()
                .app
                .windows
                .get(0)
                .cloned()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "missing window config"))?;

            let data_dir = std::env::temp_dir().join("vectorta_optimizer").join("webview2");

            tauri::WebviewWindowBuilder::from_config(app.handle(), &window_cfg)?
                .data_directory(data_dir)
                .build()?;

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            load_price_data,
            list_sample_csvs,
            pick_csv_file,
            pick_save_csv,
            run_double_ma_optimization,
            compute_double_ma_drilldown,
            cancel_double_ma_optimization,
            list_moving_averages
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
