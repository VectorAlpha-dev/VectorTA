# VectorTA Documentation Style Guide

This guide ensures consistency across all documentation pages on the VectorTA website.

## Voice and Tone

### General Principles
- **Clear and accessible**: Write for developers who may not be financial experts
- **Professional but friendly**: Not overly formal, but not too casual
- **Direct and concise**: Get to the point quickly
- **Educational**: Assume the reader wants to learn, not just reference

### Person and Address
- Use second person ("you") when giving instructions
- Use first person plural ("we") when speaking as VectorTA
- Avoid passive voice when possible

✅ Good: "You can calculate the RSI by passing price data to the function"
❌ Avoid: "The RSI can be calculated by passing price data to the function"

## Language Standards

### Spelling and Grammar
- Use American English spelling (optimize, not optimise)
- Use Oxford commas in lists
- Spell out acronyms on first use: "Relative Strength Index (RSI)"

### Technical Terms
- Define technical terms on first use
- Link to glossary or external resources for complex concepts
- Use consistent terminology throughout (don't switch between "period" and "length")

### Code References
- Use backticks for inline code: `sma()`, `period = 20`
- Use language-specific code blocks for examples
- Always specify the language for syntax highlighting

## Content Structure

### Page Template
Every indicator page should follow this structure:

1. **Header**: Indicator name and one-line description
2. **Overview**: 2-3 paragraphs explaining what the indicator is and why it matters
3. **Interpretation**: How to read and use the indicator
4. **Calculation**: Mathematical explanation (keep it accessible)
5. **Parameters**: Table format with defaults and ranges
6. **Returns & Output**: What the function returns
7. **Example Usage**: Code examples with comments
8. **Common Use Cases**: 3-4 practical applications
9. **Edge Cases & Errors**: Common pitfalls and error handling
10. **References**: Links to further reading

### Headings
- Use sentence case for headings: "Common use cases" not "Common Use Cases"
- Keep headings concise and descriptive
- Use H2 for main sections, H3 for subsections

## Writing Examples

### Code Examples
```rust
// ✅ Good: Clear, commented, and runnable
use vectorta::indicators::rsi;

// Calculate 14-period RSI
let prices = vec![45.15, 46.23, 45.04, 46.08, 47.13];
let rsi_values = rsi(&prices, 14)?;

// Check for oversold condition
if let Some(rsi) = rsi_values.last() {
    if *rsi < 30.0 {
        println!("RSI indicates oversold condition: {:.2}", rsi);
    }
}
```

### Descriptions
✅ Good: "The Moving Average Convergence Divergence (MACD) shows the relationship between two moving averages of prices. Traders use it to identify changes in the strength, direction, momentum, and duration of a trend."

❌ Avoid: "MACD calculation involves EMA subtraction for trend analysis purposes."

### Parameter Documentation
✅ Good:
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `period` | 14 | 2-100 | Number of periods for calculation. Lower values are more sensitive to price changes. |

❌ Avoid: "period: integer (default 14)"

## Common Patterns

### Explaining Complexity
- Always include big-O notation for algorithms
- Explain in plain language what this means

Example: "Complexity: O(n) - The calculation time grows linearly with the amount of data. Processing 1,000 data points takes roughly 10x longer than 100 points."

### Error Descriptions
Format errors consistently:
- **Error name**: Brief description
- **When it occurs**: Specific conditions
- **How to fix**: Actionable solution

Example:
- **NotEnoughData**: The indicator requires more historical data than provided
- **When it occurs**: Called with fewer than `period + 1` data points
- **How to fix**: Ensure your data array has at least 15 elements for a 14-period RSI

### Linking
- Link to other indicators when mentioned
- Link to external resources sparingly but helpfully
- Use descriptive link text, never "click here"

✅ Good: "RSI works well with [Bollinger Bands](/indicators/bbands) for confirmation"
❌ Avoid: "For more info on Bollinger Bands, [click here](/indicators/bbands)"

## Formatting Guidelines

### Lists
- Use bullet points for unordered information
- Use numbered lists for sequential steps
- Keep list items parallel in structure

### Tables
- Use tables for parameter documentation
- Include units where applicable
- Align numbers to the right, text to the left

### Emphasis
- Use **bold** for important terms or warnings
- Use *italics* sparingly for emphasis
- Use `code` formatting for any code elements

### Notes and Warnings
Use consistent formatting for special callouts:

```markdown
> **Note**: Interactive charts require WASM support in your browser.

> **Warning**: This indicator may produce false signals in ranging markets.

> **Tip**: Combine with volume indicators for better accuracy.
```

## Review Checklist

Before publishing any documentation:

- [ ] Follows the standard page template
- [ ] All technical terms are defined
- [ ] Code examples are tested and working
- [ ] Parameters are documented in table format
- [ ] Common use cases are included
- [ ] Error conditions are explained
- [ ] Links are working and descriptive
- [ ] Grammar and spelling are correct
- [ ] Tone is consistent with style guide

## Quick Reference

### Dos
- Do explain the "why" not just the "what"
- Do include practical examples
- Do use simple language first, technical details second
- Do test all code examples
- Do link to related indicators

### Don'ts
- Don't assume expert knowledge
- Don't use unexplained jargon
- Don't write walls of text
- Don't forget edge cases
- Don't use passive voice excessively

## Maintenance

This style guide is a living document. As the library grows and user feedback comes in, we'll update these guidelines to better serve our users.