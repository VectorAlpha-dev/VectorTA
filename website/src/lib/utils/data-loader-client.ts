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