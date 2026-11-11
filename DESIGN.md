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
    fontFamily: "Host Grotesk, Helvetica Neue, Arial, sans-serif"
    fontSize: "clamp(2.7rem, 5.4vw, 4.6rem)"
    fontWeight: 600
    lineHeight: 0.97
    letterSpacing: "-0.04em"
  headline:
    fontFamily: "Host Grotesk, Helvetica Neue, Arial, sans-serif"
    fontSize: "clamp(2.2rem, 3.6vw, 3.1rem)"
    fontWeight: 600
    lineHeight: 1.02
    letterSpacing: "-0.035em"
  title:
    fontFamily: "Host Grotesk, Helvetica Neue, Arial, sans-serif"
    fontSize: "1.45rem"
    fontWeight: 600
    lineHeight: 1.15
    letterSpacing: "-0.022em"
  body:
    fontFamily: "Source Serif 4, Iowan Old Style, Georgia, serif"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: 1.68
  label:
    fontFamily: "Host Grotesk, Helvetica Neue, Arial, sans-serif"
    fontSize: "0.75rem"
    fontWeight: 500
    lineHeight: 1
  mono:
    fontFamily: "Fragment Mono, SFMono-Regular, Menlo, monospace"
    fontSize: "0.64rem"
    fontWeight: 400
    lineHeight: 1.4
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

The Executed Paper makes a causal model read as an inspectable research publication. Navy typography, source-style figures, and quiet hairlines establish scholarly precision; aqua and periwinkle blooms make the white ground feel observed rather than inert. Evidence is always present as executed output: a world field, a plotted plate, a decomposition, or a measured reading position.

The system is intentionally airy and terse. It uses one display voice for structure, a serif reading voice for explanation, and mono only where the reader is inspecting code, measurements, or data labels. White and soft-white surfaces are separated by rules instead of card piles; color is concentrated in figures, focused states, and the active peach marker.

Light and dark schemes preserve the same publication hierarchy. Dark mode uses navy as the field and lifts plates tonally rather than changing the information model. Motion gives the real generated world and its explanatory sequence a single continuous instrument, while reduced-motion readers receive the same evidence without animation.

**Key Characteristics:**
- Executed figures and the site share a single publication palette and data-label face; causal graphs are the exception and follow Graphviz (see Do's).
- Hairline rails and node chains make navigation a reading instrument.
- Peach identifies the current, selected, or magnified point of attention.
- Plates remain flat; depth belongs to the loupe and the floating resume mark.
- The responsive composition preserves evidence first, with the story stage moving ahead of its prose on narrow screens.

## Colors

A white publication ground is held by navy ink, with aqua and periwinkle used as atmospheric fields and data families; peach is reserved for attention.

### Primary
- **Deep Navy Ink** (`{colors.navy}`): anchors all display type, rules, graph outlines, footer ground, and the primary action.
- **Publication Link Blue** (`{colors.link-blue}`): carries text links, code keywords, direct-treatment series, and the primary-action hover state.

### Secondary
- **Pastel Aqua** (`{colors.aqua}`): forms selection, field bloom, positive/advisory washes, covariate family marks, and the aqua portion of the shared figure palette.

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

**Display Font:** Host Grotesk (with Helvetica Neue, Arial, sans-serif fallbacks)
**Body Font:** Source Serif 4 (with Iowan Old Style, Georgia, serif fallbacks)
**Label/Mono Font:** Fragment Mono (with SFMono-Regular, Menlo, monospace fallbacks)

**Character:** The grotesk creates concise, tightly tracked scientific headings and interface labels; the optical serif gives explanatory paragraphs a book-like reading voice. Fragment Mono is an evidentiary face for seeds, coefficients, code, figure labels, and versions, never generic decoration.

### Hierarchy
- **Display** (600, `clamp(2.7rem, 5.4vw, 4.6rem)`, 0.97): home hero statement only, held to a 10.5ch measure.
- **Headline** (600, `clamp(2.2rem, 3.6vw, 3.1rem)`, 1.02): documentation H1 at a 20ch maximum, with balanced wrapping.
- **Title** (600, 1.45rem, 1.15): H2 section breaks, preceded by generous separation and a hairline rule.
- **Body** (400, 0.875rem, 1.68): reading text, constrained to the 66ch publication measure; the H1 lead rises to 1.05rem at 1.55.
- **Label** (500, 0.75rem, 1): buttons and compact interface actions.
- **Data label** (400, 0.64rem, 1.4): seeds, versions, captions, figure measurements, and navigation metadata in Fragment Mono.

### Named Rules
**The Three Voices Rule.** Host Grotesk structures, Source Serif 4 explains, and Fragment Mono measures. Mono is reserved for code, data, and measurement.

## Layout

The site uses a 70rem page grid, a 3.2rem single-row header, and a 66ch text measure. Reading pages open under a shallow aqua/periwinkle field, then settle into white paper; the left section index and right contents rail become connected node chains at desktop widths. The right rail begins at 60em, and the primary navigation’s rail treatment begins at 76.25em.

The home hero occupies the viewport beneath the header, bounded by a 34rem minimum and 56rem maximum. Its evidence field fills the background while a 34rem copy block stays at the left. The explanatory story uses a 0.78fr / 1.22fr two-column grid with a sticky stage; below 59.984375em, the stage moves ahead of the prose and the hero becomes a 50rem composition. Index links use a responsive grid with 16rem minimum tracks, not equal-height cards.

**The Evidence Column Rule.** Explanatory text stays within the 66ch measure; full-width evidence plates and the home field may exceed it when the data needs room to be inspected.

## Elevation & Depth

This is a flat publication system. Hairline rules, quiet tonal fields, and light/dark plate changes establish hierarchy; ordinary cards, code plates, figures, rails, and notes have no shadow. The only lifted instruments are the magnifiers and a floating resume mark, whose shadows communicate temporary proximity rather than permanent surface hierarchy.

### Shadow Vocabulary
- **Floating resume** (`0 12px 32px -12px rgba(12, 31, 64, 0.28)` in light; `0 14px 36px -12px rgba(0, 0, 0, 0.55)` in dark): raises the saved reading location only when the contents rail is unavailable.
- **Figure loupe** (`0 0 0 1px var(--pg-navy), 0 16px 34px -10px rgba(12, 31, 64, 0.38)`): separates the circular figure inspection lens from the plate beneath it.
- **Hero loupe** (28px blur with a 10px vertical offset): gives the cursor-following world lens its single physical lift.

### Named Rules
**The Lens-Only Lift Rule.** Do not add ambient shadows to publication plates. Depth is reserved for an instrument that temporarily magnifies or resumes reading.

## Shapes

Plates are restrained rounded rectangles: code, figures, notes, and resume marks use a 0.35rem corner; the live story stage uses 0.5rem. Search fields are 0.6rem when opened. Actions, node labels, status dots, the figure loupe, and causal-world nodes are fully circular or pill-shaped. Borders are typically 1px or 1.2px navy-derived hairlines, and the current node is a peach circle outlined in navy.

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
- **Typography:** Fragment Mono at 0.7rem with 1.7 line height and ligatures disabled.

### Cards / Containers
- **Corner Style:** executed-figure plates use the 0.35rem plate corner; live story stages use 0.5rem.
- **Background:** figures are white paper; note and advisory containers use transparent-bordered aqua, periwinkle, or peach washes.
- **Shadow Strategy:** flat by default; use the Lens-Only Lift Rule for inspection and resume controls only.
- **Border:** 1px navy-derived hairline; tables use stronger top and bottom rules instead of a box.
- **Internal Padding:** figure plates use 1.1rem × 1.2rem × 0.9rem; code plates use 1rem × 1.2rem.

### Inputs / Fields
- **Style:** the search control is a pill with white ground and a 1px inset hairline; its interface face is Host Grotesk at 0.72rem.
- **Focus:** it opens over the right side of the header at desktop width; the global Link Blue focus outline remains visible for keyboard use.

### Navigation
- **Style:** the Atlas-style header is one 3.2rem row: wordmark, section links, search, version, scheme switch, and repository action. No second tab bar is shown.
- **States:** section links are muted ink at rest and navy on hover or active state; active sections gain a small peach dot with a navy outline. The left and right reading rails repeat that connected-node grammar, filling completed stations with navy and the current station with peach.
- **Mobile treatment:** section links yield to the menu below 76.25em; the version label hides below 44.984375em and the repository pill keeps only its icon.

### Executed Figure Plate
- **Character:** a numbered evidence plate, not an illustration card.
- **Behavior:** each `scm-plot` increments a Plate counter and places a mono plate label before its serif caption. Fine-pointer readers can inspect the figure through a 184px circular loupe at 2.4×; the clone is the original SVG or image, not a blurred raster enlargement.

### Chapter Chain and Resume Mark
- **Character:** a reading-progress instrument made of stations on a hairline.
- **Behavior:** the current chapter and contents station enlarge and become peach; passed stations fill navy. Dots drift vertically by 1.5px only when reduced motion is not requested. After a reader leaves a page, the optional resume mark restores its saved heading or scroll position.

### Executed World Field
- **Character:** a full-bleed causal world used as evidence at the start of the publication.
- **Behavior:** a canvas draws the build-time world with three blurred ghost-depth layers, aqua/periwinkle blooms, family-colored edges, a sales ridge, and a cursor-following 2.2× circular lens. The lens exposes node roles and coefficients; the time cursor follows the pointer. The code-to-series-to-graph-to-decomposition-to-identity story uses the same world and morphs on scroll, with a zero-duration fallback for reduced motion.

## Do's and Don'ts

### Do:
- **Do** use the same navy/aqua/periwinkle/peach figure language in package plots, executed notebooks, and site plates.
- **Do** draw causal graphs in Graphviz's manner, as PyMC's `model_to_graphviz` does: top-to-bottom ranks, black ovals labelled `name ~ type`, plain black arrows, a Times-like serif (STIX), and grey fills by observability (white for latent factors and the baseline, gray90 controls, gray80 channels, gray70 outcome). Dashes mark isolated nulls and the baseline edge; no other colour.
- **Do** keep explanations in the serif reading column and reserve Fragment Mono for code, seeds, coefficients, captions, and other measurements.
- **Do** make a focused or current causal datum peach, with a navy outline where the mark needs to remain precise.
- **Do** preserve both light and dark publication schemes; change the ground and plate tone, not the information hierarchy.
- **Do** honor `prefers-reduced-motion` by retaining the world and proof while removing loupe transitions, chain drift, and story interpolation.

### Don't:
- **Don't** introduce colors outside the PyMC Labs navy, aqua, periwinkle, soft-white, and peach families or their tints and shades.
- **Don't** use mono as a display substitute; it is for code, measurements, and figure data labels.
- **Don't** turn ordinary documentation content into shadowed cards; use white plates, tonal washes, and hairlines instead.
- **Don't** use peach as generic decoration; reserve it for the active, selected, inspected, or outcome datum.
- **Don't** create a second figure style for notebooks or package output; executed evidence must look like it belongs to the same publication.
