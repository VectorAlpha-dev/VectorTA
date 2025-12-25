#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod state;
mod commands;

use commands::*;
use state::AppState;

fn main() {
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

            let data_dir = std::env::temp_dir()
                .join("ta_desktop_demo")
                .join("webview2");

            tauri::WebviewWindowBuilder::from_config(app.handle(), &window_cfg)?
                .data_directory(data_dir)
                .build()?;

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![load_price_data, run_double_ma_optimization])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
