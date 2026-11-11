---
name: pymc-generator
description: "An executed research publication for causal-world documentation."
colors:
  navy: "#0c1f40"
  aqua: "#b4e7dd"
  periwinkle: "#9faae2"
  soft-white: "#f7f7f7"
  peach: "#f6ae72"
  paper: "#ffffff"
  link-blue: "#3d57b0"
  muted-ink: "#3b4a66"
  dusk-plate: "#11284d"
  mist-ink: "#f2f4f8"
typography:
  display:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "clamp(2.3rem, 4.4vw, 3.8rem)"
    fontWeight: 600
    lineHeight: 1.02
    letterSpacing: "-0.02em"
  headline:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "clamp(1.8rem, 2.9vw, 2.5rem)"
    fontWeight: 600
    lineHeight: 1.08
    letterSpacing: "-0.02em"
  title:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "1.25rem"
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "-0.01em"
  body:
    fontFamily: "Inter, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: 1.65
  label:
    fontFamily: "Inter, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "0.75rem"
    fontWeight: 500
    lineHeight: 1
  mono:
    fontFamily: "Fragment Mono, SFMono-Regular, Menlo, monospace"
    fontSize: "0.7rem"
    fontWeight: 400
    lineHeight: 1.7
rounded:
  code: "0.2rem"
  plate: "0.35rem"
  stage: "0.5rem"
  search: "0.6rem"
  pill: "999px"
spacing:
  xs: "0.4rem"
  sm: "0.6rem"
  md: "0.9rem"
  lg: "1.4rem"
  xl: "1.8rem"
  section: "3.4rem"
components:
  button-secondary:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.navy}"
    typography: "{typography.label}"
    rounded: "{rounded.pill}"
    padding: "0.7rem 1.25rem"
  button-primary:
    backgroundColor: "{colors.navy}"
    textColor: "{colors.paper}"
    typography: "{typography.label}"
    rounded: "{rounded.pill}"
    padding: "0.7rem 1.25rem"
  repository-link:
    backgroundColor: "{colors.navy}"
    textColor: "{colors.paper}"
    typography: "{typography.mono}"
    rounded: "{rounded.pill}"
    padding: "0.42rem 0.8rem 0.42rem 0.9rem"
  code-plate:
    backgroundColor: "{colors.soft-white}"
    textColor: "{colors.navy}"
    typography: "{typography.mono}"
    rounded: "{rounded.plate}"
    padding: "1rem 1.2rem"
  figure-plate:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.navy}"
    rounded: "{rounded.plate}"
    padding: "1.1rem 1.2rem 0.9rem"
  atlas-navigation:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.navy}"
  search-field:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.navy}"
    rounded: "{rounded.pill}"
  chapter-chain:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.navy}"
  resume-mark:
    backgroundColor: "{colors.soft-white}"
    textColor: "{colors.navy}"
    rounded: "{rounded.plate}"
    padding: "0.5rem 0.6rem"
---

# Design System: pymc-generator

## Overview

**Creative North Star: "The Executed Paper"**

The Executed Paper makes a causal model read as an inspectable research publication. Navy typography, source-style figures, and quiet hairlines establish scholarly precision; aqua and periwinkle blooms make the white ground feel observed rather than inert. The home page opens on one abstract figure, a network world in a single ink; past it, evidence is always present as executed output: the demo window's code, files, and data, a plotted plate, a decomposition, or a measured reading position.

The system is intentionally airy and terse. It follows the PyMC Labs type pairing: Archivo Expanded for headings and large text, Inter for reading, captions, and the interface, and a monospace only where the reader is reading code. White and soft-white surfaces are separated by rules instead of card piles; color is concentrated in figures, focused states, and the active peach marker.

Light and dark schemes preserve the same publication hierarchy. Dark mode uses navy as the field and lifts plates tonally rather than changing the information model. Motion belongs to the home figure and the demo window, while reduced-motion readers receive a still figure, the demo's final frame, and the same evidence without animation.

**Key Characteristics:**
- Executed figures and the site share a single publication palette and the same two faces (Inter labels, Archivo Expanded titles); causal graphs are the exception and follow Graphviz (see Do's).
- Hairline rails and node chains make navigation a reading instrument.
- Peach identifies the current, selected, or magnified point of attention.
- Plates remain flat; depth belongs to the loupe and the floating resume mark.
- The responsive composition preserves evidence first, with the demo window stacking its file list above its pane on narrow screens.

## Colors

A white publication ground is held by navy ink, with aqua and periwinkle used as atmospheric fields and data families; peach is reserved for attention.

### Primary
- **Deep Navy Ink** (`{colors.navy}`): anchors all display type, rules, graph outlines, footer ground, and the primary action.
- **Publication Link Blue** (`{colors.link-blue}`): carries text links, code keywords, direct-treatment series, and the primary-action hover state.

### Secondary
- **Pastel Aqua** (`{colors.aqua}`): forms selection, field bloom, the dark-scheme ink of the home figure, positive/advisory washes, covariate family marks, and the aqua portion of the shared figure palette.

### Tertiary
- **Soft Periwinkle** (`{colors.periwinkle}`): supplies atmospheric blooms, code washes, treatment-family marks, and periwinkle figure ramps.
- **Peach Orange** (`{colors.peach}`): marks the active rail or section node, the outcome, loupe focus, and other current-state signals.

### Neutral
- **Soft White** (`{colors.soft-white}`): is the light-scheme plate and footer type color.
- **Paper** (`{colors.paper}`): is the white reading ground and executed-figure plate.
- **Muted Ink** (`{colors.muted-ink}`): carries secondary reading text and light-scheme figure captions.
- **Dusk Plate** (`{colors.dusk-plate}`): lifts dark-scheme plates from the navy field.
- **Mist Ink** (`{colors.mist-ink}`): is the high-contrast dark-scheme text and rule color.

### Named Rules
**The Focused Peach Rule.** Peach names the current or inspected datum: active navigation, current progress nodes, the outcome node, and the time cursor. It is not a general surface fill.

## Typography

**Display Font:** Archivo at its expanded width (`wdth` 125, the brand's Archivo Expanded), with Helvetica Neue, Arial, sans-serif fallbacks. The site loads only the expanded width, so every Archivo run sets expanded without per-rule `font-stretch`.
**Body and Interface Font:** Inter (with system-ui, -apple-system, Segoe UI, sans-serif fallbacks), optical sizes 14–32.
**Code Font:** Fragment Mono (with SFMono-Regular, Menlo, monospace fallbacks), for code alone; the brand names no monospace.

**Character:** Archivo Expanded gives headings a wide, confident presence, so they are set smaller and with lighter tracking than a condensed grotesk would need. Inter carries everything a reader reads or operates: paragraphs, captions, plate labels, buttons, navigation, tables, and sheet figures (with tabular numerals). Code, notebook cells, and their outputs stay monospaced so columns of code align.

### Hierarchy
- **Display** (Archivo Expanded 600, `clamp(2.3rem, 4.4vw, 3.8rem)`, 1.02, −0.02em): home hero statement only, held to a 10.5ch measure.
- **Headline** (Archivo Expanded 600, `clamp(1.8rem, 2.9vw, 2.5rem)`, 1.08, −0.02em): documentation H1 at a 20ch maximum, with balanced wrapping. Home section headings use `clamp(1.4rem, 2.3vw, 2rem)`.
- **Title** (Archivo Expanded 600, 1.25rem, 1.2): H2 section breaks, preceded by generous separation and a hairline rule; H3 0.98rem, H4 0.82rem.
- **Body** (Inter 400, 0.875rem, 1.65): reading text, justified across the full content column with automatic hyphenation, so its edges meet those of code plates, equations, figures, and rules; inline code never takes a hyphen, and below 30em the text runs ragged. The H1 lead rises to 1.05rem at 1.55.
- **Label** (Inter 500, 0.6–0.78rem, 1): buttons, navigation, plate labels, chapter titles, versions, and table heads; numerals tabular where they align.
- **Code** (Fragment Mono 400, 0.7rem, 1.7, ligatures off): code plates, inline code, notebook cells and outputs.

### Named Rules
**The Two Faces Rule.** Archivo Expanded heads, Inter reads and labels. A monospace appears only inside code; it is never a label, caption, or measurement face.

## Layout

The site uses a 70rem page grid, a 3.2rem single-row header, and one content column shared by text and evidence. Reading pages open under a shallow aqua/periwinkle field, then settle into white paper; the left section index and right contents rail become connected node chains at desktop widths. The right rail begins at 60em, and the primary navigation’s rail treatment begins at 76.25em.

The home hero occupies the viewport beneath the header, bounded by a 34rem minimum and 56rem maximum. A 34rem copy block stays at the left; the network figure sits in the space to its right and fades out toward the copy and the hero's top and bottom. Below 59.984375em the hero sizes to its copy with the figure in a band of up to 24rem beneath it. The demo section that follows fits one screen beneath the header: text on the left (heading, lead, chapters) and the window on the right in a `minmax(14rem, 0.62fr) / 1.38fr` grid, over a colour field that spans the viewport. Below 74.984375em the text sits above the window and the chapter list yields. The pane height follows the viewport: `clamp(19rem, 100svh − 14rem, 30rem)` beside the text, `clamp(18rem, 100svh − 21rem, 27rem)` stacked, `clamp(18rem, 100svh − 17rem, 26rem)` in a narrow window; below 29.984375em the lead yields too. Index links use a responsive grid with 16rem minimum tracks, not equal-height cards.

**The Evidence Column Rule.** Explanatory text, code plates, equations, and evidence plates share one column and both of its edges; text is justified to meet them rather than held to a narrower measure.

## Elevation & Depth

This is a flat publication system. Hairline rules, quiet tonal fields, and light/dark plate changes establish hierarchy; ordinary cards, code plates, figures, rails, and notes have no shadow. The only lifted surfaces are the magnifiers, a floating resume mark, and the home demo window; their shadows communicate temporary proximity or a staged screen rather than permanent surface hierarchy.

### Shadow Vocabulary
- **Floating resume** (`0 12px 32px -12px rgba(12, 31, 64, 0.28)` in light; `0 14px 36px -12px rgba(0, 0, 0, 0.55)` in dark): raises the saved reading location only when the contents rail is unavailable.
- **Figure loupe** (`0 0 0 1px var(--pg-navy), 0 16px 34px -10px rgba(12, 31, 64, 0.38)`): separates the circular figure inspection lens from the plate beneath it.
- **Hero lens** (28px blur with a 10px vertical offset): gives the home figure's magnifying lens its single physical lift.
- **Demo window** (`0 0 0 1px var(--pg-rule), 0 40px 80px -36px rgba(12, 31, 64, 0.42)` in light; the same with `rgba(0, 0, 0, 0.6)` in dark): floats the home editor over its colour field, as a screen recording would.

### Named Rules
**The Lens-Only Lift Rule.** Do not add ambient shadows to publication plates. Depth is reserved for an instrument that temporarily magnifies or resumes reading, and for the home demo window, which is a staged screen rather than a plate.

## Shapes

Plates are restrained rounded rectangles: code, figures, notes, resume marks, and the demo's explorer rows use a 0.35rem corner; the home demo window uses 0.5rem. Search fields are 0.6rem when opened. Actions, node labels, status dots, the figure loupe, and causal-world nodes are fully circular or pill-shaped. Borders are typically 1px or 1.2px navy-derived hairlines, and the current node is a peach circle outlined in navy.

## Components

### Buttons
- **Character:** compact, pill-shaped reading actions rather than large promotional controls.
- **Shape:** full pill (`{rounded.pill}`).
- **Primary:** navy ground, paper text, and 0.7rem × 1.25rem padding; hover changes only to Publication Link Blue.
- **Secondary:** paper ground with a 1px inset hairline; hover strengthens the hairline to navy.
- **Focus:** the shared 2px Link Blue outline sits 3px outside the control.

### Code Plates
- **Character:** source and result are paired pieces of evidence.
- **Shape:** 0.35rem corner with a 1px hairline.
- **Background:** soft-white plate for source; white ground for executed result.
- **Typography:** Fragment Mono at 0.7rem with 1.7 line height and ligatures disabled; the only place a monospace appears.

### Cards / Containers
- **Corner Style:** executed-figure plates use the 0.35rem plate corner; the home demo window uses 0.5rem.
- **Background:** figures are white paper; note and advisory containers use transparent-bordered aqua, periwinkle, or peach washes.
- **Shadow Strategy:** flat by default; use the Lens-Only Lift Rule for inspection and resume controls only.
- **Border:** 1px navy-derived hairline; tables use stronger top and bottom rules instead of a box.
- **Internal Padding:** figure plates use 1.1rem × 1.2rem × 0.9rem; code plates use 1rem × 1.2rem.

### Inputs / Fields
- **Style:** the search control is a pill with white ground and a 1px inset hairline; its interface face is Inter at 0.72rem.
- **Focus:** it opens over the right side of the header at desktop width; the global Link Blue focus outline remains visible for keyboard use.

### Navigation
- **Style:** the Atlas-style header is one 3.2rem row: wordmark, section links, search, version, scheme switch, and repository action. No second tab bar is shown.
- **States:** section links are muted ink at rest and navy on hover or active state; active sections gain a small peach dot with a navy outline. The left and right reading rails repeat that connected-node grammar, filling completed stations with navy and the current station with peach. Each rail is headed by a small Archivo Expanded 600 title (0.74rem, navy: the section name, "On this page") on the page's own ground, with no sticky white band; stations sit inside the rail's clipping box so every dot draws whole, centred on its hairline.
- **Mobile treatment:** section links yield to the menu below 76.25em; the version label hides below 44.984375em and the repository pill keeps only its icon.

### Executed Figure Plate
- **Character:** a numbered evidence plate, not an illustration card.
- **Behavior:** each `scm-plot` increments a Plate counter and places an Inter plate label before its caption. Fine-pointer readers can inspect the figure through a 184px circular loupe at 2.4×; the clone is the original SVG or image, not a blurred raster enlargement.

### Source Viewer
- **Character:** an API page's source sits one click away without breaking the reading column.
- **Row:** each "Source code in `path`" box is a single hairline-bordered row on the page's ground: a muted code mark at the left, Inter 500 at 0.68rem, the file path as inline code, and an expand mark at the right; the border darkens on hover.
- **Dialog:** the row opens a modal dialog (min(72rem, 100vw − 3rem) by min(100svh − 3rem, 64rem); nearly full-screen on phones) over the page, which is blurred (7px) under a navy veil and stops scrolling. A header names the object and its file (Archivo Expanded 600 name, mono path) beside a round close button; below it the code scrolls on its own, in the plate tone, and takes focus so arrow and page keys move it. Line numbers scroll the code to their line without touching the page. Escape, the close button, or a click on the blurred page closes it and returns focus to the row; the code block moves into the dialog and back, so its copy button keeps working. Reduced motion drops the rise-and-fade entrance. Without the script the row expands in place.

### Chapter Chain and Resume Mark
- **Character:** a reading-progress instrument made of stations on a hairline.
- **Behavior:** the current chapter and contents station enlarge and become peach; passed stations fill navy. Dots drift vertically by 1.5px only when reduced motion is not requested. After a reader leaves a page, the optional resume mark restores its saved heading or scroll position.

### Home Network Figure
- **Character:** one network in one ink (navy on white, aqua on navy) and the only image in the first viewport. It is an abstract figure, not build-time data; executed evidence begins in the demo beneath it.
- **Rendering:** a canvas draws the network at a third of the screen's resolution (3px cells), ordered-dithers it with a 4×4 Bayer matrix to four ink tones over a transparent ground, and scales it up without smoothing. Depth sets each node's size and tone; strokes thin on small figures.
- **Bodies:** at rest the nodes cover a slowly turning globe, each joined to its three nearest neighbours, over a faint dithered body. Clicking the globe, or the "Break the world apart" pill beneath it, snaps the edges, bursts the nodes outward, and lands them on a flowing ribbon lattice that tapers toward the copy (2.8s); after 9s, or on another click ("Gather it back"), the ribbon gathers back into the globe near end first (7s). Every edge runs from a lower to a higher node index, so both bodies are acyclic.
- **Lens:** a 96px circle at 2.2× draws the same network at full resolution with an arrow on each edge. It follows the pointer and otherwise travels from node to node; it is not drawn below 600px.
- **Reduced motion:** the globe holds still, the lens moves only with the pointer, and the pill is hidden.

### Home Demo Window
- **Character:** a minimal editor that plays like a short screen recording in two parts. One world: a notebook samples it, writes its bundle, and opens `dataset.csv` as a sheet. Many worlds: the same notebook draws a twelve-world corpus, saves it as one compressed `worlds.npz`, loads it back, opens the archive to list the arrays inside, and opens `outcome_raw.npy` with one row per world. The window is illustrative and labelled so; its contents are not.
- **Contents:** `scm_docs.editor_demo` executes the four notebook cells during the build, in order, in a temporary directory, and shows what each displays as Jupyter would (the `repr` of a final expression). The explorer lists the files those cells wrote; the dataset sheet shows the first 24 rows of the bundle's `dataset.csv`; the archive view lists every member of `worlds.npz` with its shape, type, and compressed size; the worlds sheet shows the first 16 weeks of every world in `outcome_raw`. Status lines carry the true counts and sizes.
- **Composition:** traffic-light dots in rule grey (no off-palette hues), tabs that accumulate as files open, a kernel chip, a 13.5rem explorer, and a pane that holds the notebook, a sheet, or the archive. Opening a sheet or the archive folds the explorer away, so the data takes the whole window; sheets stretch to the window's width, and the worlds sheet shares its rows across the pane's height. Code uses the site's highlight colours in Fragment Mono; sheets are set in Inter with tabular figures, lettered columns and numbered rows for the CSV, world and week indices for the array, and a Link Blue selection. Behind the window, aqua, periwinkle, and an aqua/Link Blue mix drift as soft radial fields, masked to fade into the page above and below.
- **Text and chapters:** the heading, lead, and a chapter list sit left of the window. The six chapters (Sample a world, Write its bundle, Open the data, Draw a corpus, Load it back, Look inside the file) are stations on a hairline in the site's chain grammar: the current one peach and enlarged, passed ones navy. Each jumps the demo to its start.
- **Playback:** it plays once, the first time 45% of the window is on screen, in about 33s: typing at 16ms per character, each cell running with `[*]` and a busy kernel dot, files appearing in the explorer as they are written, and a cursor that travels to and clicks each thing it opens, remembering where it left off. Leaving the view pauses it; returning resumes it unless the reader paused it. A pill under the window reads Pause, Play, or Replay, and a 2px line along the window's foot shows progress. Every frame is derived from one clock, so pausing, replaying, and jumping to a chapter need no other state.
- **Fitting:** code scales with the window so its 49-character longest line always fits (`min(0.68rem, (100cqi − 21.5rem) / 30)` beside the explorer; `min(0.48rem, 3.4cqi − 3px)` in a narrow window). Below a 40rem window width the explorer becomes a two-column list above the pane and the lights and kernel chip hide. When the pane is shorter than the notebook, the notebook keeps its latest cell in view, as a real one scrolls.
- **Reduced motion and no script:** the page ships the final frame (the worlds sheet open, every file listed, all four cells run). Without the script it stays; with reduced motion nothing plays and the pill stays hidden, but each chapter shows that chapter's last frame as a still.

## Do's and Don'ts

### Do:
- **Do** use the same navy/aqua/periwinkle/peach figure language in package plots, executed notebooks, and site plates.
- **Do** draw causal graphs in Graphviz's manner, as PyMC's `model_to_graphviz` does: top-to-bottom ranks, black ovals labelled `name ~ type`, plain black arrows, a Times-like serif (STIX), and grey fills by observability (white for latent factors and the baseline, gray90 controls, gray80 channels, gray70 outcome). Dashes mark isolated nulls and the baseline edge; no other colour.
- **Do** keep explanations in Inter in the reading column, headings in Archivo Expanded, and a monospace for code alone.
- **Do** make a focused or current causal datum peach, with a navy outline where the mark needs to remain precise.
- **Do** preserve both light and dark publication schemes; change the ground and plate tone, not the information hierarchy.
- **Do** honor `prefers-reduced-motion` by holding the home figure still, showing the demo as still frames, and removing loupe transitions and chain drift.

### Don't:
- **Don't** introduce colors outside the PyMC Labs navy, aqua, periwinkle, soft-white, and peach families or their tints and shades.
- **Don't** use a monospace for labels, captions, or measurements, or Archivo Expanded for running text.
- **Don't** turn ordinary documentation content into shadowed cards; use white plates, tonal washes, and hairlines instead.
- **Don't** use peach as generic decoration; reserve it for the active, selected, inspected, or outcome datum.
- **Don't** create a second figure style for notebooks or package output; executed evidence must look like it belongs to the same publication.
