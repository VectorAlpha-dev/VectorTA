use std::collections::HashMap;
use std::sync::Mutex;

use my_project::utilities::data_loader::{read_candles_from_csv, Candles};

#[derive(Default)]
pub struct AppState {
    pub candles: Mutex<HashMap<String, Candles>>, 
}

impl AppState {
    pub fn load_price_data(&self, path: &str) -> Result<String, String> {
        let candles = read_candles_from_csv(path).map_err(|e| e.to_string())?;
        let mut map = self
            .candles
            .lock()
            .map_err(|_| "failed to lock candles map".to_string())?;
        let id = format!("data-{}", map.len() + 1);
        map.insert(id.clone(), candles);
        Ok(id)
    }

    pub fn get_candles(&self, id: &str) -> Result<Candles, String> {
        let map = self
            .candles
            .lock()
            .map_err(|_| "failed to lock candles map".to_string())?;
        map.get(id)
            .cloned()
            .ok_or_else(|| format!("unknown data id: {id}"))
    }
}
