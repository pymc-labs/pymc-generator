# Pinned reusable-family backend (internal)

This is an **opt-in experimental backend**, not a replacement for
`DataGenerator.generate` or the library's default random-stream contract.

Source: this repository's research commit
`d61f98f596c5de885947eb5144897ed0909e7c5d`:

- `benchmarks/study_200k_methods.py` → `template.py`
- relevant helpers/base adapter from `benchmarks/comparison_methods.py` → `helpers.py`
- `config` and `make_cell` from `benchmarks/study_200k_run.py` → `inputs.py`
- `benchmarks/study_200k_packing.py` → `packing.py`

The files are included here so production launches **do not depend on `/tmp`, another
worktree, a research checkout, dynamic source-string patches, or changing `sys.path`**.
They reuse the existing library's scientific constructors. Function cloning changes
only named collaborators; it does not monkeypatch the global scientific modules.

Adaptations from that pinned source:

- relative imports and formatting; remove unused benchmark helpers and RNG diagnostic
  replay infrastructure (not the primary sampler);
- 208 reported weeks instead of 104;
- defer train/validation assignment (`is_val=2`) **before normalization**, retaining
  temporal support/query masks and per-world support-scale semantics;
- clear per-candidate audit histories between chunks in `FamilyGenerator`; keep the
  model and compiled draw function alive;
- bounded candidate top-up in the orchestration wrapper, with explicit shortfalls;
- mechanism priors (reference contributions, shape ranges, log-uniform Michaelis-Menten
  scale), built by the library's template helpers: their per-slot reports are cropped to
  the active slots, as the corpus schema keeps inactive slots at zero;
- texture flags (hf, pulse) at inclusion probability 0 or 1 and schedule flags at 0 are
  template constants, not graph inputs: `make_cell` sets them, the graph samples the
  rest (a schedule flag at probability 1 is sampled, and is always 1);
- a retryable numeric draw failure (the library's predicate) leaves its candidate
  without a world and counts as a draw failure instead of stopping the worker; a task
  none of whose candidates draws stops the worker with the last error.

The composable prior (`inputs.config`) has maximum 15 treatment slots, 10 controls and
one latent slot. Active counts are 5–15 treatments and 3–10 controls. Active masks
affect causal generation, not merely the saved columns. A family pair is constant
within a worker; counts, causal structure, smoothness and component flags vary per
world without recompiling. Component flags are sampled inside the graph.
Parameters/innovations remain fresh.

The original compositional prior is retained. Signed controls/contributions are
allowed; this is **not** the dashboard demo's positive-controls prior, and it enables no
mechanism priors, so its saturation-family arrays are absent: the dashboard's split
chart may be empty even though the family pair is recorded in each completion's
provenance. A structural-split design (`StreamPlan.design`) compiles the legacy member
configurations instead (`generation.DESIGN_PRESETS`, mechanism priors included) and
rejects worlds with a negative control. Random-walk truth recording is not supported.

This backend intentionally uses private generator APIs and a copied packing routine.
A library change requires replay, schema/readback and scientific invariant tests;
source digests refuse silently resuming a run with changed implementation. Do not
claim same-seed equivalence to the ordinary generator.
