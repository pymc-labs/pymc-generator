# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Researchers and ML engineers who need synthetic causal data with known ground
truth. Mainly people building or evaluating marketing-mix and causal-effect
estimators, including amortized or prior-fitted models trained on generated
corpora. Secondary: PyMC / PyTensor users auditing single generated worlds.

## Product Purpose

`pymc-generator` draws random Pearlian structural causal models and compiles
each into one PyTensor / PyMC graph, so a single draw yields the observable
series and the exact additive decomposition of the outcome (identity error
around 1e-15), reproducibly from a seed. It writes human-auditable bundles
(CSVs, description, DAG, figures, replay recipe) and stacked, padded tensor
corpora with explicit active counts and masks.

## Positioning

Exact, replayable generating truth rather than an approximation, stated with
honest limits: known truth does not imply that effects are identifiable from
observations or recoverable by any particular fitted model.

## Operating Context

Readers arrive from GitHub, papers, or PyMC Labs material, then work through
the guides, API reference, and executed notebooks while writing Python in their
own environment. The documentation site is MkDocs Material, deployed to GitHub
Pages from `main`.

## Capabilities and Constraints

- Five node families (latent, covariates, treatments, baseline, outcome), eight
  edge types, carryover x saturation response curves, eight trajectory
  components, Gaussian identity-link outcome only.
- Every figure and number on the documentation site is produced by executing
  code during the site build (markdown-exec blocks and mkdocs-jupyter
  notebooks); nothing is pasted in. Notebook execution stays coupled to the
  site build.
- Figures rendered by the package (`pymc_generator.viz`, bundle PNGs) and by the
  executed notebooks share one visual style with the site, so the site and its
  generated outputs are maintained together.
- Alpha software: public APIs and persistence contracts may change.

## Brand Commitments

- Name `pymc-generator`; published by PyMC Labs; MIT licensed.
- No AI-native or LLM-targeted sections (no llms.txt, "copy for LLM" controls,
  or machine-readable summaries aimed at language models).
- Colour stays within the PyMC Labs brand palette (Brand Identity Guidelines
  2025): Deep Navy Blue `#0C1F40` and Pastel Aqua `#B4E7DD` (main), Soft
  Periwinkle `#9FAAE2` and Soft White `#F7F7F7` (secondary), Peach Orange
  `#F6AE72` (accent). Tints and shades of these hues are allowed; radically
  different hues are not.

## Evidence on Hand

- Build-time executed worlds and corpora via `docs_gen/scm_docs.py`.
- Notebooks in `docs/examples/` (end-to-end tour, MMM recovery case study,
  dataset typicality).
- No testimonials, adoption figures, or benchmark claims exist; none may be
  invented.

## Product Principles

1. Show the truth, executed: every claim on the site is backed by output the
   build actually computed.
2. Precision over persuasion: state what the generator does and its limits
   exactly.
3. One source of style: the site, notebooks, and package figures look like one
   publication.
4. Restraint: dense information without clutter; crowding many things together
   is a failure, not richness.
