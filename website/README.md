# Technical Analysis Indicators Documentation Website

This is the documentation website for the Rust-based technical analysis library, featuring 100+ indicators with interactive charts and comprehensive documentation.

## Features

- 📊 **Interactive Charts**: Real-time charting with TradingView Lightweight Charts
- 🎛️ **Adjustable Parameters**: Live parameter controls for each indicator
- 🌙 **Dark Mode**: Built-in dark mode support
- 🔍 **Search**: Fast client-side search (Ctrl/Cmd + K)
- 📱 **Responsive**: Works on all device sizes
- ⚡ **Fast**: Built with Astro's Islands architecture
- 🦀 **WASM Ready**: Prepared for WebAssembly integration

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

This will:
1. Scan indicators from the parent Rust project
2. Generate documentation pages
3. Cache sample data
4. Start the development server at http://localhost:4321

### Build

```bash
npm run build
```

The built site will be in the `dist/` directory.

### Project Structure

```
website/
├── src/
│   ├── components/     # React components for charts and UI
│   ├── content/        # Generated MDX content for indicators
│   ├── data/          # Indicator registry
│   ├── layouts/       # Astro layouts
│   ├── lib/           # Utilities and mock WASM
│   ├── pages/         # Astro pages
│   └── styles/        # Global styles
├── scripts/           # Build scripts
└── public/           # Static assets
```

## Key Scripts

- `npm run scan-indicators` - Scans Rust source for indicators
- `npm run generate-pages` - Generates MDX pages for each indicator
- `npm run cache-data` - Caches sample OHLCV data
- `npm run prebuild` - Runs all preparation scripts

## Adding New Indicators

1. Add the indicator to the Rust project
2. Run `npm run scan-indicators` to update the registry
3. Run `npm run generate-pages` to create the documentation page
4. The indicator will automatically appear in the navigation

## WASM Integration

The project uses mock implementations for now. To integrate real WASM:

1. Build the Rust project with `wasm-pack`
2. Replace `MockTALib` in `src/lib/mock-wasm/index.ts`
3. Update the `calculate` method to call real WASM functions

## Deployment

The site can be deployed to any static hosting service:

- Netlify: `npm run build` and deploy the `dist/` folder
- Vercel: Connect your GitHub repo
- GitHub Pages: Use the Astro GitHub Pages action

## License

Same as the parent Rust project.