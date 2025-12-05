#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod state;
mod commands;

use commands::*;
use state::AppState;

fn main() {
    tauri::Builder::default()
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![load_price_data, run_double_ma_optimization])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
