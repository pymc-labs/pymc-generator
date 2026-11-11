---
template: home.html
title: Simulate causal worlds
hide:
  - navigation
  - toc
  - footer
---

<section class="pg-demo" data-pg-demo aria-labelledby="pg-demo-title" markdown>
<div class="pg-demo__field" aria-hidden="true"><i></i><i></i><i></i></div>
<div class="pg-demo__text">
<h2 id="pg-demo-title">One world, or a whole corpus.</h2>
<p class="pg-demo__lead">Write one world as a folder you can open, its data beside the true decomposition and graph. Or draw many and save them as one compressed <code>.npz</code>, every world padded to a common shape.</p>
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
