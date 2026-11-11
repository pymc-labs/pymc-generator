---
template: home.html
title: Simulate causal worlds
hide:
  - navigation
  - toc
  - footer
---

```python exec="1" html="1"
from scm_docs import world, world_json

print(world_json(world(scenario=1, seed=0), seed=0))
```

<section class="pg-story" data-pg-story aria-label="One draw, from code to proof">
<div class="pg-story__steps">
<div class="pg-story__step" data-step="code">
<h2>One call draws a world.</h2>
<p>A named scenario's prior and a seed.</p>
</div>
<div class="pg-story__step" data-step="series">
<h2>It returns the data a modeller sees.</h2>
<p>Spend, controls, and sales, week by week.</p>
</div>
<div class="pg-story__step" data-step="graph">
<h2>And the graph that caused it.</h2>
<p>Hidden demand drives both spend and sales.</p>
</div>
<div class="pg-story__step" data-step="decomposition">
<h2>Every week splits into its causes.</h2>
<p>Baseline, each channel, and indirect effects add back to sales.</p>
</div>
<div class="pg-story__step" data-step="identity">
<h2>Exactly.</h2>
<p>The largest gap between sales and the sum of its parts.</p>
</div>
</div>
<div class="pg-story__stage">
<svg class="pg-stage" data-pg-stage viewBox="0 0 800 560" role="img" aria-label="The drawn world: its code, series, graph, and decomposition"></svg>
<ol class="pg-story__progress" aria-hidden="true">
<li data-step="code">Code</li>
<li data-step="series">Data</li>
<li data-step="graph">Graph</li>
<li data-step="decomposition">Causes</li>
<li data-step="identity">Proof</li>
</ol>
</div>
</section>

<section class="pg-proof" markdown>

## Every figure here is executed.

```python exec="1" html="1"
from scm_docs import world, viz_html
import pymc_generator as pg

w = world(scenario=1, seed=0)
print(viz_html(pg.viz.plot_decomposition, w, caption="The same world, rendered by <code>pg.viz.plot_decomposition</code> while this page was built. Hover to read it under the loupe."))
```

</section>

<section class="pg-index" markdown>

## Read on

- [Getting started](getting-started.md)
- [The causal model](guide/foundation.md)
- [Sampling a world](guide/worlds.md)
- [The exact decomposition](guide/decomposition.md)
- [Generating a corpus](guide/corpus.md)
- [End-to-end tour](examples/index.ipynb)
- [API reference](reference/index.md)

</section>
