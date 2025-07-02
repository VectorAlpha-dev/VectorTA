#!/usr/bin/env python3
"""Fix specific bracket patterns causing compilation errors."""

import os
import re
import glob

def fix_bracket_patterns(filepath):
    """Fix common bracket patterns in a single file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern 1: Fix extra closing braces before "} else {"
    # This pattern: "}\n}\n} else {" should be "} else {"
    content = re.sub(r'\n\s*}\n\s*}\n(\s*}\s+else\s*{)', r'\n\1', content)
    
    # Pattern 2: Fix if-else blocks with extra braces
    # Match patterns like:
    # something
    #     }
    # } else {
    # And replace with:
    # something
    # } else {
    pattern = re.compile(r'(\n\s+)\}\n\}(\s+else\s*\{)', re.MULTILINE)
    content = pattern.sub(r'\1}\2', content)
    
    # Pattern 3: Fix specific wasm block pattern
    # Where we have extra } before else
    pattern = re.compile(r'(\s+)\}\n\s*\}\n(\s*\}\s+else\s*\{)', re.MULTILINE)
    content = pattern.sub(r'\1}\n\2', content)
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    """Process all indicator files."""
    fixed_files = []
    
    # Get all .rs files
    all_files = []
    all_files.extend(glob.glob("src/indicators/**/*.rs", recursive=True))
    all_files.extend(glob.glob("src/indicators/*.rs"))
    
    for filepath in all_files:
        try:
            if fix_bracket_patterns(filepath):
                fixed_files.append(filepath)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    print(f"Fixed {len(fixed_files)} files")
    if fixed_files:
        for f in sorted(set(fixed_files)):
            print(f"  {f}")

if __name__ == "__main__":
    main()