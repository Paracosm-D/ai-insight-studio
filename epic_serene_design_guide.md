# Epic + Serene Streamlit Design Guide

## Design target
Create a page that feels cinematic, calm, spacious, and high-end:
- vast, dark, atmospheric background
- restrained accent colors
- slow movement instead of flashy movement
- strong hierarchy with generous whitespace
- glassmorphism only in moderation

## Palette
- Background: `#07111F`
- Surface: `rgba(12, 24, 44, 0.58)`
- Border: `rgba(180, 214, 255, 0.16)`
- Primary text: `#EAF2FF`
- Secondary text: `#9FB0C8`
- Accent cyan: `#89D5FF`
- Accent lavender: `#A89BFF`
- Glow: `rgba(116, 190, 255, 0.18)`

## Layout
1. Hero first, controls second.
2. Use one cinematic hero section with a left text block and a right quiet data-status block.
3. Keep each analysis module to one main chart + one secondary chart + one compact table + one AI brief.
4. Add more vertical breathing room than you think you need.

## Motion
- slow background pulse: 8s to 16s
- very subtle float: 6px to 10px vertical shift
- avoid spinning, flickering, bouncing

## Typography
- English display: Inter / Manrope
- Chinese UI: Noto Sans SC / PingFang SC fallback
- Headlines: semi-bold, letter-spacing 0.02em
- Body: lighter weight and slightly dimmed

## What to implement in Streamlit
1. Inject CSS with a radial-gradient background and glow layer.
2. Replace plain title/caption with a custom HTML hero.
3. Wrap metrics, upload box, data health, and AI brief in custom cards.
4. Standardize Plotly charts with transparent backgrounds.
5. Reduce noisy default Streamlit chrome.

