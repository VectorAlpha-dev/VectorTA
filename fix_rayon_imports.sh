#!/bin/bash

# Find all .rs files in src/indicators that use rayon
echo "Finding all files that import rayon..."
FILES=$(grep -l "^use rayon::prelude::\*;" src/indicators/*.rs src/indicators/moving_averages/*.rs 2>/dev/null)

echo "Found $(echo "$FILES" | wc -w) files to fix"

for file in $FILES; do
    echo "Fixing $file..."
    
    # Replace unguarded rayon import with conditional one
    sed -i 's/^use rayon::prelude::\*;$/#[cfg(not(target_arch = "wasm32"))]\nuse rayon::prelude::*;/' "$file"
done

echo "Done! All rayon imports are now conditionally compiled."