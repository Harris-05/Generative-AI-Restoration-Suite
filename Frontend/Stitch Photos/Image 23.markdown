---
name: Generative AI Restoration Studio
colors:
  surface: '#0f131d'
  surface-dim: '#0f131d'
  surface-bright: '#353944'
  surface-container-lowest: '#0a0e18'
  surface-container-low: '#171b26'
  surface-container: '#1c1f2a'
  surface-container-high: '#262a35'
  surface-container-highest: '#313540'
  on-surface: '#dfe2f1'
  on-surface-variant: '#bcc9c6'
  inverse-surface: '#dfe2f1'
  inverse-on-surface: '#2c303b'
  outline: '#879391'
  outline-variant: '#3d4947'
  surface-tint: '#6bd8cb'
  primary: '#6bd8cb'
  on-primary: '#003732'
  primary-container: '#29a195'
  on-primary-container: '#00302b'
  inverse-primary: '#006a61'
  secondary: '#c3c0ff'
  on-secondary: '#1d00a5'
  secondary-container: '#3626ce'
  on-secondary-container: '#b3b1ff'
  tertiary: '#4edea3'
  on-tertiary: '#003824'
  tertiary-container: '#00a572'
  on-tertiary-container: '#00311f'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#89f5e7'
  primary-fixed-dim: '#6bd8cb'
  on-primary-fixed: '#00201d'
  on-primary-fixed-variant: '#005049'
  secondary-fixed: '#e2dfff'
  secondary-fixed-dim: '#c3c0ff'
  on-secondary-fixed: '#0f0069'
  on-secondary-fixed-variant: '#3323cc'
  tertiary-fixed: '#6ffbbe'
  tertiary-fixed-dim: '#4edea3'
  on-tertiary-fixed: '#002113'
  on-tertiary-fixed-variant: '#005236'
  background: '#0f131d'
  on-background: '#dfe2f1'
  surface-variant: '#313540'
typography:
  headline-xl:
    fontFamily: Inter
    fontSize: 2rem
    fontWeight: '600'
    lineHeight: 2.5rem
    letterSpacing: -0.025em
  headline-xl-mobile:
    fontFamily: Inter
    fontSize: 1.5rem
    fontWeight: '600'
    lineHeight: 2rem
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 1.5rem
    fontWeight: '600'
    lineHeight: 2rem
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Inter
    fontSize: 1.25rem
    fontWeight: '600'
    lineHeight: 1.75rem
    letterSpacing: -0.015em
  body-lg:
    fontFamily: Inter
    fontSize: 1rem
    fontWeight: '400'
    lineHeight: 1.5rem
  body-md:
    fontFamily: Inter
    fontSize: 0.875rem
    fontWeight: '400'
    lineHeight: 1.25rem
  body-sm:
    fontFamily: Inter
    fontSize: 0.75rem
    fontWeight: '400'
    lineHeight: 1rem
  code-lg:
    fontFamily: JetBrains Mono
    fontSize: 0.875rem
    fontWeight: '500'
    lineHeight: 1.25rem
    letterSpacing: -0.01em
  code-sm:
    fontFamily: JetBrains Mono
    fontSize: 0.75rem
    fontWeight: '500'
    lineHeight: 1rem
    letterSpacing: 0em
  label-md:
    fontFamily: Inter
    fontSize: 0.8125rem
    fontWeight: '500'
    lineHeight: 1.125rem
    letterSpacing: 0.01em
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 0.6875rem
    fontWeight: '500'
    lineHeight: 0.875rem
    letterSpacing: 0.04em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  gutter: 1rem
  gutter-desktop: 1.5rem
  margin: 1rem
  margin-desktop: 1.5rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-lg: 1rem
  space-xl: 1.5rem
---

## Brand & Style
The design system embodies the rigor, precision, and clarity of an advanced computer vision and machine learning laboratory. Tailored for deep learning researchers, optical restoration engineers, and imaging scientists, the interface eliminates decorative friction in favor of high-throughput data inspection, comparative visual analysis, and deterministic model configuration.

The aesthetic balances clean, functional minimalism with refined scientific-instrument styling. It uses structured grid-based instrumentation, high-density telemetry displays, and subtle optical separation to create an environment that feels computational, reliable, and laser-focused on perceptual fidelity.

## Colors
The palette is built around an instrument-grade dark slate canvas (`#0B0F19`) engineered to minimize eye strain and maximize dynamic range when inspecting high-resolution image restorations and pixel-level loss maps. An alternative off-white canvas (`#F8FAFC`) serves bright-room benchwork environments.

- **Primary (`#0D9488` / `#14B8A6`)**: Spectral Teal. Used for primary model execution actions, active inspection tools, and highlighted viewport modes.
- **Secondary (`#4F46E5`)**: Deep Indigo. Applied to hyperparameter controls, secondary checkpoint badges, and specialized diffusion pipeline toggles.
- **Tertiary / Success (`#10B981`)**: Precision Emerald. Reserved for system telemetry, inference health, cluster status indicators ("Backend: connected"), and statistical benchmark gains (PSNR/SSIM improvements).
- **Signal & Alert**: Amber (`#F59E0B`) indicates GPU VRAM thermal throttling and tensor quantization warnings. Rose (`#F43F5E`) designates out-of-memory errors, gradient clipping exceptions, and non-convergent runs.
- **Neutrals**: Layered zinc/slate values (`#0B0F19`, `#111827`, `#1E293B`, `#334155`) structure viewport panels, dividing borders, and baseline cards.

## Typography
Typography is organized around a dual-type architecture:
1. **Inter** handles structural user interaction, viewport titles, metadata descriptions, and control surface labels. Its optical neutrality ensures effortless skimming across dense layouts.
2. **JetBrains Mono** anchors all raw inference data, numerical telemetry, tensor shapes, execution latency metrics, and benchmark logs.

Tabular figures (`tnum`) must be explicitly enabled on all JetBrains Mono instances to prevent visual jitter during live model streaming and parameter scrubbing.

## Layout & Spacing
The layout follows a modular multi-panel workbench model. Content adapts fluidly to ultrawide scientific displays while maintaining structural rigidity across high-density research toolboxes.

- **Grid Model**: A balanced 12-column variable workspace that reflows into a 3-column split-panel workbench on desktop: Model Configuration & Layer Controls (3 cols), Comparative Viewport Center (6 cols), and Statistical Telemetry / Metrics (3 cols).
- **Responsive Adaptations**:
  - **Desktop (>= 1280px)**: Persistent dual/quad viewport split with synchronized pan-and-zoom and real-time inference telemetry docks.
  - **Tablet (768px - 1279px)**: Collapsible sidebar drawers, stacked before/after comparison via an interactive center split slider.
  - **Mobile (< 768px)**: Tabbed single-panel view (Inspect, Config, Metrics) with swipe-based image crossfades.

## Elevation & Depth
Visual hierarchy is established using layered flat surfaces and sharp, low-contrast structural outlines rather than heavy atmospheric drop shadows:

- **Base Canvas**: Dark slate `#0B0F19` sits at z-index ground zero.
- **Surface Containers (Cards & Panels)**: Elevated to `#111827` using a subtle 1px border (`#1E293B` in dark mode, `#E2E8F0` in light mode).
- **Active / Focused Overlays**: Hovered panels and selected node states elevate via micro-borders of `#0D9488` at 30% alpha combined with an ultra-subtle ambient glow (`0 0 20px -5px rgba(13, 148, 136, 0.15)`).
- **Floating HUDs & Loupes**: Viewport overlays, zoom loupe widgets, and context inspectors utilize translucent glassmorphism with `backdrop-filter: blur(12px)` and 80% opacity surface fills (`rgba(17, 24, 39, 0.8)`).

## Shapes
A unified curvature scale balances technical sharpness with modern ergonomic comfort:
- **Panels & Cards**: Built with `rounded-xl` (1.5rem) corners on primary workspace wrappers and `rounded-lg` (1rem) on nested inspect sub-cards.
- **Micro-Controls**: Buttons, toggle switches, and input fields utilize standard `rounded` (0.5rem) geometries to retain an engineered, mechanical profile.
- **Badges & Status Pills**: Retain fully rounded pill borders (`rounded-full`) to immediately differentiate operational states from configurable rectangular controls.

## Components

### Buttons & Interactive Controls
- **Primary Inference Action**: Solid Spectral Teal fill (`#0D9488`), white Inter text (`label-md`), sharp transition states with micro-ring highlights (`focus:ring-2 focus:ring-teal-500/40`).
- **Secondary Tools**: Outlined button with `#1E293B` border, transparent background, and `#94A3B8` icon/text label.
- **Step / Stepper Toggles**: Segmented pill bars with active state pill in `#1E293B` surface and accent text.

### Viewports & Comparative Panels
- **Split Comparison Canvas**: Dual-pane canvas containing original vs. restored tensors with an interactive center divider handle. Features synchronized hardware-accelerated zoom loupe (1x to 32x pixel grid view).
- **Image Drop-Zone**: Dashed 1.5px border (`border-dashed border-slate-700`), darkened well fill, interactive drop state shifts to solid `#0D9488` with subtle inset glow.

### Telemetry Badges & Status Indicators
- **Backend Connection Pill**: Inline pill with an animated glowing dot (`#10B981`), `code-sm` font showing latency (`34ms`) and device (`CUDA:0 RTX 4090`).
- **Evaluation Metric Chips**: Monospace key-value blocks (e.g., `PSNR: +4.2 dB`, `SSIM: 0.964`) with delta indicators colored green for gains and rose for degradations.

### Form Inputs & Hyperparameter Sliders
- **Numeric Fields**: Integrated step controls with monospace numerical displays and inline units (`px`, `steps`, `cfg`).
- **Diffusion Sliders**: Slim 4px track (`#1E293B`) with a teal thumb (`#14B8A6`) and live-updating numerical tooltip displaying exact float precision (`0.001` step).

### Statistical Benchmark Charts & Probability Bars
- **Residual Loss & Confidence Histograms**: Low-profile sparklines with discrete vertical bars for token probabilities. Outlined with `#1E293B` container lines, utilizing gradient fills from Indigo (`#4F46E5`) to Teal (`#0D9488`).