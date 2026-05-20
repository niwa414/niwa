# mietclaw Design System

## 1. Product position

mietclaw should feel like a **scientific operations console**, not a chatbot.

The visual language should communicate:
- control
- traceability
- computational depth
- long-running scientific workflows
- confidence under failure and recovery

A useful shorthand is:

> "Mission control for multiscale materials simulation."

## 2. Brand direction

### Name lockup
Use **mietclaw** as the primary product wordmark.

- `miet` = clean, precise, engineered
- `claw` = orchestration, reach, control plane

### Brand voice
- calm, exact, technical
- never playful in critical workflow surfaces
- confident without sounding like a generic AI copilot
- explain results as scientific operations, not chat replies

### Tagline
> Operate MD-to-KMC materials simulations from one control plane.

### UI reference direction
The front-end should now feel closer to a **Codex-style operator shell**:
- dark, quiet, editor-grade surfaces
- sharper typography and denser information layout
- less glossy dashboard chrome
- more agent workspace, fewer “marketing hero” gestures

## 3. Visual concept

The design should combine three metaphors:

1. **Lattice** — crystalline order, spatial structure
2. **Energy landscape** — barriers, transitions, rates
3. **Operations console** — queues, status, checkpoints, recovery

That means the UI should prefer:
- grids
- layered panels
- sharp information hierarchy
- restrained motion
- glowing accents used sparingly to indicate live computation

## 4. Color system

### Core palette
- `Obsidian`: `#07111F`
- `Deep Panel`: `#0D1728`
- `Raised Panel`: `#13213A`
- `Edge`: `#223455`
- `Text Primary`: `#EAF2FF`
- `Text Secondary`: `#9FB3D9`

### Signal colors
- `Lattice Teal`: `#54E1C1`
- `Field Blue`: `#7B8CFF`
- `Barrier Amber`: `#FFB347`
- `Recovery Red`: `#FF6B6B`

### Usage rule
The base UI should stay dark and stable. Bright colors are for:
- active stages
- barrier/rate emphasis
- warnings
- recovery state
- selected objects

## 5. Typography

### Primary
Use a clean modern sans stack:
- Inter
- ui-sans-serif
- system-ui

### Secondary / data
Use monospace for:
- file paths
- energy barriers
- rates
- state labels
- CLI snippets

Recommended stack:
- JetBrains Mono
- ui-monospace
- SFMono-Regular
- Menlo

## 6. Layout rules

### Main shell
Three-column app shell:
- left: navigation + project scope
- center: job creation / pipeline / execution timeline
- right: explanation / recovery / artifacts

### Spacing rhythm
Use a tight engineering rhythm:
- 6px micro spacing
- 12px control spacing
- 18px card padding baseline
- 28px section separation

### Corners and strokes
- radius: 14px for panels
- radius: 999px for pills
- borders should be thin and low-contrast
- surfaces should separate primarily by luminance, not thick lines

## 7. Motion

Motion should feel like instrumentation, not marketing.

Allowed motion:
- subtle gradient drift in hero background
- status pulse for active stages
- hover lift of <= 2px
- short 180–220ms transitions

Avoid:
- bouncy animations
- exaggerated parallax
- fake AI sparkles

## 8. Component language

### Status chips
States should be obvious at a glance:
- queued = slate
- running = teal/blue
- completed = green-teal
- failed = red
- recovery available = amber

### Pipeline stages
Each stage card should show:
- stage name
- tool/engine
- status
- key output
- rerun/resume affordance

### Explanation cards
Should summarize:
- what was computed
- what was derived from it
- what downstream stage consumed it
- what to inspect if it failed

## 9. Data visualization style

Charts should feel analytical and sober.

Use:
- dark plot backgrounds
- thin lines
- highlighted current series
- annotated thresholds
- no rainbow palette

Primary visualizations to support later:
- barrier distributions
- Arrhenius relationships
- jump frequency trends
- KMC time evolution
- artifact lineage

## 10. Frontend MVP pages

The first branded frontend should ship these surfaces:

1. landing / product overview
2. job console
3. run timeline
4. artifact archive
5. explanation panel

## 11. Anti-patterns

Do not make it look like:
- a generic SaaS KPI dashboard
- a consumer AI chat product
- a biotech pastel UI
- a gaming neon interface

## 12. Design summary

If someone opens the app for five seconds, they should think:

> "This is a serious control room for materials simulation workflows."
