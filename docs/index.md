---
template: home.html
title: "pymc-generator: Open Source Structural Causal Model Simulation with Exact Decompositions in Python"
description: >-
  pymc-generator is an open-source Python library from PyMC Labs that generates
  structural causal models and simulates time series from them, with the exact
  decomposition of the outcome as ground truth.
hide:
  - navigation
  - toc
  - footer
---

<section class="pg-about" aria-labelledby="pg-about-title" markdown>
<div class="pg-about__main" markdown>

## What is pymc-generator? { #pg-about-title }

<p class="pg-about__statement"><code>pymc-generator</code> is an open-source Python library from PyMC Labs that generates structural causal models (SCMs) and simulates data from them with known ground truth.</p>

It generates synthetic time series together with the exact decomposition of the
outcome into baseline, direct treatment contributions and indirect effects. Its
default shape is the marketing mix model (MMM): treatments with carryover and
saturation, observed covariates and hidden confounders. The machinery itself is
not specific to marketing.

<div class="pg-about__blocks" markdown>
<div class="pg-about__block" markdown>

### How pymc-generator works

`pymc-generator` draws a random Pearlian directed acyclic graph over latent
factors, covariates, treatments, a baseline and one outcome, then compiles it
into a single PyMC and PyTensor graph. One draw from that graph returns the
observable series and the true components that produced them, reproducibly
from a seed. A treatment's direct effect on the outcome passes through a
carryover (adstock) and saturation response curve from `pymc-marketing`.

</div>
<div class="pg-about__block" markdown>

### What pymc-generator produces

The true components sum to the outcome exactly, to about 1e-15 in float64. A
single world is written as a folder a person can audit: the dataset and its
decomposition as CSV, a plain-text description, the causal graph and figures.
Many worlds are stacked into one compressed `.npz` corpus, padded to a common
shape with masks for the active inputs.

</div>
<div class="pg-about__block" markdown>

### What pymc-generator is used for

`pymc-generator` is used to check causal-effect estimators, including marketing
mix models, against data whose true decomposition is known, and to generate
corpora for training amortized models. The
[MMM recovery case study](examples/simple-model.ipynb) fits two marketing mix
models to one simulated world and reports the actual error and coverage.

</div>
<div class="pg-about__block" markdown>

### What pymc-generator does not claim

Known generating truth does not imply that effects are identifiable from
observations or recoverable by any particular fitted model. The outcome
likelihood is Gaussian with an identity link. `pymc-generator` is alpha
software.

</div>
</div>
</div>
<aside class="pg-about__facts" aria-label="Project facts" markdown>
<dl>
<dt>Published by</dt><dd>PyMC Labs</dd>
<dt>License</dt><dd>MIT, open source</dd>
<dt>Language</dt><dd>Python 3.13 or newer</dd>
<dt>Built on</dt><dd>PyMC, PyTensor, pymc-marketing</dd>
<dt>Status</dt><dd>Alpha</dd>
</dl>

[View on GitHub](https://github.com/pymc-labs/pymc-generator){ .pg-button }

</aside>
</section>

<section class="pg-demo" data-pg-demo aria-labelledby="pg-demo-title" markdown>
<div class="pg-demo__field" aria-hidden="true"><i></i><i></i><i></i></div>
<div class="pg-demo__text">
<h2 id="pg-demo-title">Generate one synthetic dataset, or a whole training corpus</h2>
<p class="pg-demo__lead">Write one simulated world as a folder you can open: the dataset as CSV, beside its true decomposition, causal graph and figures. Or draw many worlds and save them as one compressed <code>.npz</code> corpus, with every world padded to a common shape.</p>
<ol class="pg-demo__chapters" aria-label="Demo chapters">
<li><button type="button" data-pg-chapter>Sample a world</button></li>
<li><button type="button" data-pg-chapter>Write its bundle</button></li>
<li><button type="button" data-pg-chapter>Open the data</button></li>
<li><button type="button" data-pg-chapter>Draw a corpus</button></li>
<li><button type="button" data-pg-chapter>Load it back</button></li>
<li><button type="button" data-pg-chapter>Look inside the file</button></li>
</ol>
</div>
<div class="pg-demo__stage" markdown>

```python exec="1" html="1"
from scm_docs import editor_demo

print(editor_demo(scenario=1, seed=0))
```

<div class="pg-demo__meta"><span>Illustrative demo. The code, files, and data come from this page's build.</span><button class="pg-demo__play" type="button" data-pg-play hidden>Pause</button></div>
</div>
</section>

<section class="pg-index" aria-labelledby="pg-index-title" markdown>

## Explore the pymc-generator documentation { #pg-index-title }

<div class="pg-cards pg-cards--lead" markdown>

[**Getting started** Install the package, sample one world, read its description, check the decomposition, and write an auditable bundle.](getting-started.md){ .pg-card }

[**End-to-end tour** An executed notebook that builds a world one knob at a time, from the smallest graph to a confounded world.](examples/index.ipynb){ .pg-card }

</div>

<div class="pg-cards" markdown>

[**The causal model** The five families of nodes, and what a modeller observes versus what confounds them.](guide/foundation.md){ .pg-card }

[**Sampling a world** What one accepted world contains, and how to reach every piece of it.](guide/worlds.md){ .pg-card }

[**The exact decomposition** How every unit of outcome was produced, summing to the outcome exactly.](guide/decomposition.md){ .pg-card }

[**Generating a corpus** Many worlds stacked into one padded tensor archive, with the decomposition targets included.](guide/corpus.md){ .pg-card }

[**MMM recovery case study** Two marketing mix models and the generator's reference model, fitted to the same simulated observations.](examples/simple-model.ipynb){ .pg-card }

[**API reference** Every public function and class, generated from the source docstrings.](reference/index.md){ .pg-card }

</div>
</section>

<section class="pg-install" aria-labelledby="pg-install-title" markdown>
<div class="pg-install__text" markdown>

## Install pymc-generator { #pg-install-title }

`pymc-generator` installs from its repository with [uv](https://docs.astral.sh/uv/)
and the checked-in dependency lock. It requires Python 3.13 or newer.

[Get started](getting-started.md){ .pg-button .pg-button--primary }
[Read the model](guide/foundation.md){ .pg-button }

</div>
<div class="pg-install__code" markdown>

```bash
git clone https://github.com/pymc-labs/pymc-generator.git
cd pymc-generator
uv sync --frozen
uv run --no-sync python
```

</div>
</section>
