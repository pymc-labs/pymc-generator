# pymc-generator

Synthetic marketing-mix-model (MMM) datasets with known structural ground truth.
Each world contains media spend, observed controls, latent demand, and sales,
plus an additive, intervention-based decomposition of the generated outcome.

The generator uses **PyMC** distributions, a **PyTensor** causal graph, and
**pymc-marketing** adstock and saturation transforms. Discrete graph structures
and mechanism families are sampled before constructing each world. Outputs
support individual-world inspection and padded corpora for amortized inference.

**Alpha software.** Extracted from
[`pymc-labs/structural-pfn`](https://github.com/pymc-labs/structural-pfn).
Public API and corpus migrations are recorded in [CHANGELOG.md](CHANGELOG.md).
Known generating truth does not imply that the effects are identifiable from
observations or recoverable by every fitted MMM.

## Install with uv

Use the checked-in lockfile rather than resolving a new modeling stack:

```bash
git clone https://github.com/pymc-labs/pymc-generator.git
cd pymc-generator
uv sync --frozen
uv run --no-sync python
```

Requires Python 3.13 or newer. The lock selects released registry packages:
PyMC 6.2.0, PyTensor 3.2.4, and pymc-marketing 1.1.0; no Git development
companions are required. Dependency upgrades can change numerical results even
with the same seeds. See [getting started](docs/getting-started.md) for installation
and executable examples, and [CONTRIBUTING.md](CONTRIBUTING.md) for development
and documentation environments.

## Install with conda

Use a locally built channel, or a GitHub release's native channel archive when available:

```bash
conda create --name pymc-generator --override-channels --strict-channel-priority \
  --channel "file://$PWD/dist/conda-channel" --channel conda-forge \
  python=3.13 pymc-generator=0.0.2
conda activate pymc-generator
```

The local channel contains only `pymc-generator`; released modeling and other
native dependencies come from conda-forge, with selected source versions pinned
to match the uv lock. This is not a pip overlay and does not require publication
of `pymc-generator` on PyPI or conda-forge. See the
[conda installation guide](docs/getting-started.md#install-with-conda) for channel
paths and platform-specific environment exports, or
[native build instructions](CONTRIBUTING.md#conda-artifacts) when building from source.

## Quickstart

```python
import pymc_generator as pg

cfg = pg.make_scm_prior(
    n_treatments=3,
    n_covariates=1,
    n_latent=1,
    n_time_steps=32,
    n_cells=2,
    draws_per_cell=2,
    seed=42,
)

# One world, at the configured maximum dimensions.
world = pg.sample_scm(cfg, seed=42)
print(pg.describe_scm(world))
print("decomposition residual:", world.identity_error())

# Four worlds, with active dimensions sampled from the configuration ranges.
corpus = pg.sample_prior_predictive(cfg)
pg.save_corpus(corpus, "corpus.npz")
loaded = pg.load_corpus("corpus.npz")
print(loaded["spend_raw"].shape)  # (4, 32, 3)
```

For a CSV/figure bundle, call `pg.write_scm_bundle(world, "my-world")`.
The destination must be absent or empty. Bundles include the complete
`true_components.csv` decomposition and a `recipe.json` recording effective
configuration, seeds, environment, and replay instructions.

The command-line interface generates named inspection scenarios:

```bash
uv run --no-sync pymc-generator --out inspection-datasets --seed 20260712
```

## Data contracts and interpretation

- **Decomposition:** `sales = baseline + contributions.sum(-1) + indirect_effects`,
  up to floating-point error. Here `baseline` is the **complete non-media
  aggregate**, including observation noise, not just the structural intercept.
  The [decomposition guide](docs/guide/decomposition.md) defines the individual
  columns, floor scopes, and ordered indirect attribution.
- **Causal meaning:** direct demand/control edges enter sales, not the parentless
  intercept. A channel without a direct sales edge can still affect sales
  through downstream channels. See the [foundation](docs/guide/foundation.md).
- **Corpus storage:** arrays use configured maximum dimensions and explicit
  active counts/masks. Numerical payloads are stored as float32; generation
  identities are evaluated before that rounding. Readers migrate supported
  legacy schemas and reject malformed or unsupported metadata. See the
  [corpus guide](docs/guide/corpus.md) and [schema reference](docs/reference/corpus.md).
- **Reproducibility:** the same configuration and seed reproduce numerical
  outputs in the same locked environment. Runtime diagnostics are not
  deterministic; bitwise equality across dependency, hardware, or BLAS changes
  is not promised. Bundle recipes record the environment rather than embedding
  the software needed to recreate it.
- **Diagnostics:** signal and dependence summaries describe a generated sample.
  Optional gates are screening rules, not certificates of identification or
  successful parameter recovery. The [MMM recovery case study](docs/examples/simple-model.ipynb)
  reports actual coverage, sampling warnings, and convergence diagnostics.
  Its [oracle model](docs/guide/oracle.md) is a plug-in reference, not a universal
  lower bound on estimation error.

## Documentation and contributions

The [documentation sources](docs/index.md) are the authoritative guide and API
reference. Build the site locally with the instructions in
[CONTRIBUTING.md](CONTRIBUTING.md#documentation); notebook examples execute during
that build. Edit the notebooks directly—there are no source-generating builders.

- [World generation and configuration](docs/guide/worlds.md)
- [Inspection bundles and replay](docs/guide/bundles.md)
- [Outcome distributions](docs/reference/outcomes.md)
- [Data diagnostics](docs/guide/data-diagnostics.md)
- [Configuration reference](docs/reference/config.md)
- [API reference](docs/reference/index.md)

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a change.
Report vulnerabilities through the channels in [SECURITY.md](SECURITY.md).

## License

MIT; see [LICENSE](LICENSE). The project builds on
[PyMC](https://github.com/pymc-devs/pymc),
[PyTensor](https://github.com/pymc-devs/pytensor), and
[pymc-marketing](https://github.com/pymc-labs/pymc-marketing), which retain their
own licenses and attribution.

The MIT notice preserves Carlos Trujillo's 2024 copyright from the
`structural-pfn` extraction at commit `d5fd09f`, alongside PyMC Labs' 2026
copyright for this project.
