export interface OHLCV {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// Client-side data loader
export async function loadSampleData(): Promise<OHLCV[]> {
  try {
    const response = await fetch('/data/sample-ohlcv.json');
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error loading sample data:', error);
    return [];
  }
}

// Load CSV data
export async function loadCSVData(filename: string): Promise<OHLCV[]> {
  try {
    const response = await fetch(filename);
    const text = await response.text();
    
    // Parse CSV
    const lines = text.trim().split('\n');
    const headers = lines[0].split(',');
    
    // Find column indices
    const timeIdx = headers.findIndex(h => h.toLowerCase().includes('time') || h.toLowerCase().includes('date'));
    const openIdx = headers.findIndex(h => h.toLowerCase() === 'open');
    const highIdx = headers.findIndex(h => h.toLowerCase() === 'high');
    const lowIdx = headers.findIndex(h => h.toLowerCase() === 'low');
    const closeIdx = headers.findIndex(h => h.toLowerCase() === 'close');
    const volumeIdx = headers.findIndex(h => h.toLowerCase().includes('volume'));
    
    // Parse data
    const data: OHLCV[] = [];
    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',');
      if (values.length < headers.length) continue;
      
      data.push({
        time: parseInt(values[timeIdx]),
        open: parseFloat(values[openIdx]),
        high: parseFloat(values[highIdx]),
        low: parseFloat(values[lowIdx]),
        close: parseFloat(values[closeIdx]),
        volume: parseFloat(values[volumeIdx]) || 0,
      });
    }
    
    return data;
  } catch (error) {
    console.error('Error loading CSV data:', error);
    return [];
  }
}