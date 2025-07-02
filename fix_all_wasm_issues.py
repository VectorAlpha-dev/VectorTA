#!/usr/bin/env python3
"""Fix all WASM conditional compilation block issues."""

import os
import re
import glob

def fix_wasm_blocks_comprehensive(content):
    """Fix all WASM blocks that need closing braces."""
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a #[cfg(target_arch = "wasm32")] { line
        if re.search(r'#\[cfg\(target_arch = "wasm32"\)\]\s*\{', line):
            fixed_lines.append(line)
            i += 1
            
            # Collect the block content
            brace_count = 1
            block_lines = []
            
            while i < len(lines) and brace_count > 0:
                current_line = lines[i]
                block_lines.append(current_line)
                
                # Count braces
                brace_count += current_line.count('{')
                brace_count -= current_line.count('}')
                
                i += 1
            
            # Check if the next non-empty line contains "} else {"
            j = i
            while j < len(lines) and lines[j].strip() == '':
                j += 1
                
            if j < len(lines) and re.search(r'^\s*\}\s*else\s*\{', lines[j]):
                # We need to add a closing brace
                # Find the proper indentation
                indent_match = re.match(r'^(\s*)', lines[j])
                indent = indent_match.group(1) if indent_match else ''
                
                # Remove the last line if it's empty
                if block_lines and block_lines[-1].strip() == '':
                    block_lines.pop()
                
                # Add all block lines
                fixed_lines.extend(block_lines)
                # Add the closing brace
                fixed_lines.append(indent + '}')
            else:
                # No else block following, just add the lines as is
                fixed_lines.extend(block_lines)
        else:
            fixed_lines.append(line)
            i += 1
    
    return '\n'.join(fixed_lines)

def fix_missing_if_braces(content):
    """Fix missing closing braces in if blocks."""
    # Pattern to find incomplete if blocks
    pattern = re.compile(
        r'(\s+)if\s+[^{]+\{[^}]*?\n(\s+)\}\s*else\s*\{',
        re.MULTILINE | re.DOTALL
    )
    
    def check_braces(text):
        """Check if braces are balanced in the text."""
        count = 0
        for char in text:
            if char == '{':
                count += 1
            elif char == '}':
                count -= 1
        return count
    
    # Find all matches and check if they need fixing
    matches = list(pattern.finditer(content))
    
    # Process matches in reverse order to maintain positions
    for match in reversed(matches):
        block = match.group(0)
        # Extract the if block content
        if_start = block.find('{')
        else_pos = block.rfind('} else {')
        if_content = block[if_start:else_pos]
        
        # Check if braces are balanced
        imbalance = check_braces(if_content)
        
        if imbalance > 0:
            # Need to add closing braces
            # Find where to insert them
            lines = block[:else_pos].split('\n')
            fixed_lines = []
            
            for line in lines:
                fixed_lines.append(line)
            
            # Add the missing braces
            indent = match.group(2)
            for _ in range(imbalance):
                fixed_lines.append(indent + ' ' * 4 + '}')
            
            # Reconstruct
            fixed_block = '\n'.join(fixed_lines) + '\n' + block[else_pos:]
            content = content[:match.start()] + fixed_block + content[match.end():]
    
    return content

def main():
    """Process all indicator files."""
    fixed_files = []
    
    # Get all .rs files
    all_files = []
    all_files.extend(glob.glob("src/indicators/**/*.rs", recursive=True))
    all_files.extend(glob.glob("src/indicators/*.rs"))
    
    for filepath in all_files:
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Skip if no wasm blocks
            if '#[cfg(target_arch = "wasm32")]' not in content:
                continue
            
            original_content = content
            
            # Fix missing if braces first
            content = fix_missing_if_braces(content)
            
            # Then fix WASM blocks
            content = fix_wasm_blocks_comprehensive(content)
            
            if content != original_content:
                with open(filepath, 'w') as f:
                    f.write(content)
                fixed_files.append(filepath)
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    print(f"Fixed {len(fixed_files)} files:")
    for f in sorted(set(fixed_files)):
        print(f"  {f}")

if __name__ == "__main__":
    main()