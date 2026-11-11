---
version: 1
slug: "docs-index-md"
primary_target: "docs/index.md"
related_targets: ["overrides/main.html","docs/stylesheets/site.css","overrides/home.html","docs/stylesheets/home.css","docs/javascripts/home.js","docs_gen/scm_docs.py"]
---

# Documentation site (home + reading pages)

Scope: the whole MkDocs Material site. Visitor mode: Read (the home page opens
with a demonstrative hero, then reads as documentation).
Audience/job: causal-ML researchers and PyMC users learning to generate and
audit worlds. Proof: build-time executed worlds, figures, notebooks; no
invented claims. Constraints: PyMC Labs palette only; minimal copy; figures,
notebooks, and package plots share one style; no LLM-targeted sections.
User steer on the chosen card: more minimal, less text, Atlas-style navbar, a
full-bleed cinematic hero background that reacts to the mouse, cleaner and
whiter with more colour detail.
Hero redesign steer (2026-11-11): the first hero was too loaded. Simple,
cinematic, single colour following the site scheme, a retro low-bit image on
the right, abstract and moving; networks welcome; keep the zoom lens. The user
compared a dithered ribbon and a dithered globe and chose a merge: the globe
breaks into the ribbon on click and slowly gathers back, with a visible hint.
Rest-of-page steer (2026-11-11): scrolling should be easier. The sticky
five-step story was replaced by one screen: a blurred colour field behind a
minimal editor window (reference: a clean app window on a soft ground with an
"Illustrative demo" label) that plays once on first view: a notebook runs the
script, bundle files appear, a click opens dataset.csv as a sheet. The
Read on links stay below.
Demo follow-up (2026-11-11): text on the left, the window on the right; the
sheet uses the window's full space; the same video continues past the sheet
through the corpus API: many worlds saved to one compressed .npz, the files
inside it, and every world's data. Chapters on the left let readers jump.
Plates and type follow-up (2026-11-11): the tall multi-panel decomposition at
the end briefly became a carousel of six plots; the user then cut it, so the
home is three parts: hero, one scroll to the demo, then Read on. All type
follows the PyMC Labs sheet: Archivo Expanded for headings and large text,
Inter for body, captions, and interface; a monospace for code only; package
figures use Inter labels and Archivo Expanded titles (the DAG keeps its
Graphviz serif).

## Direction contract

THESIS: The docs read as a research publication. The home opens on one quiet,
abstract figure (a network world in a single ink) that the visitor can inspect
and break apart; executed evidence starts in the demo window beneath it. Refuses the
centred hero plus feature-card grid and the busy multi-colour field.

OWN-WORLD: Navy on white or aqua on navy, one ink per scheme in the hero;
low-resolution canvas ordered-dithered to four tones; Archivo Expanded
display, Inter reading and labels, a monospace for code only; white plates with
hairlines, a circular lens, nodes on hairline chains.

STORY: The visitor sees a turning network world, inspects it through the
lens, breaks it into a ribbon and watches it gather back; one scroll later a
notebook writes one world to disk and opens its data as a sheet, then saves
a corpus as one .npz, lists the arrays inside, and opens every world; then
the guides.

FIRST VIEWPORT: Left: display title, lead, navy "Get started" pill and a
hairline "Read the model" pill. Right: the dithered globe of nodes, fading
toward the copy and the hero's edges, with a 2.2× lens drifting node to node
or following the pointer, and a small "Break the world apart" pill beneath.
Atlas navbar on top.

FORM: The Research Publication, grounded candidate 1 (pick); seed:a43fbff0fb8bda5a

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
