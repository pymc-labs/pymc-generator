# API reference

Every page here is generated directly from the source docstrings, so it never
drifts from the code. The public surface is small and reads mathematically —
**treatments** (media channels), **covariates** (controls), and **latent**
factors (hidden confounders) size the graph; you build a prior, then draw from
it.

| Symbol | Page | Purpose |
| --- | --- | --- |
| [`make_scm_prior`](config.md#prior_generator.presets.make_scm_prior) | [Configuration](config.md) | Build a validated additive-SCM prior. |
| [`SCMPrior`](config.md#prior_generator.sampler.SCMPrior) | [Configuration](config.md) | The config: sizes, edge budgets, coefficient/noise ranges. |
| [`sample_scm`](sampling.md#prior_generator.worlds.sample_scm) / [`SCM`](sampling.md#prior_generator.worlds.SCM) | [Sampling one world](sampling.md) | Draw one accepted world with its full ground truth. |
| [`sample_prior_predictive`](corpus.md) / [`DataGenerator`](corpus.md) | [Corpus generation](corpus.md) | Generate an N-world corpus (dict of arrays). |
| [`save_corpus`](corpus.md) / [`load_corpus`](corpus.md) | [Corpus generation](corpus.md) | Compressed `.npz` persistence. |
| [`describe_scm`](describe.md) | [Descriptions](describe.md) | Plain-text description of a world. |
| [`write_scm_bundle`](bundles.md) / [`write_scenario_bundles`](bundles.md) | [Audit bundles](bundles.md) | Write auditable folders. |
| [`SCENARIOS`](scenarios.md) | [Named scenarios](scenarios.md) | Five named audit scenarios. |

## Import surface

```python
import prior_generator as pg

pg.make_scm_prior            # build a config
pg.SCMPrior                  # the config dataclass
pg.sample_scm, pg.SCM        # one world + its object
pg.describe_scm              # a world's story, as text
pg.write_scm_bundle          # a world's audit folder
pg.write_scenario_bundles    # the five-scenario inspection set
pg.SCENARIOS                 # the five named recipes
pg.sample_prior_predictive   # a corpus (dict of numpy arrays)
pg.DataGenerator             # a facade over corpus generation + validation
pg.save_corpus, pg.load_corpus
```

Public symbols are lazy-loaded, so `import prior_generator` stays light — the
heavy stack (pytensor / scipy / pandas / matplotlib) is only imported when a
symbol that needs it is first used.
