# Inferred-walk oracle integration evidence

Local checks on 2026-10-16, based on PR #66 head `60122b8`, before commit.
These are model/integration checks, **not calibration, convergence, recovery or
production throughput evidence**. No Modal jobs or source-world writes occurred.

## Environment

Python 3.13.12; PyMC 6.2.0; PyTensor 3.2.4; Nutpie 0.16.11; NumPy 2.4.6;
ArviZ 1.2.0; Xarray 2026.7.0; Numba 0.65.1. Used the existing locked environment,
with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` for numerical checks.

## Reproducible focused checks

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync pytest \
  tests/test_oracle_family.py tests/test_oracle_inferred.py \
  tests/test_oracle_dynamic.py --runslow -q
```

**64 passed in 126.41 s.** Coverage includes:

- exact mixture vs enumeration of supplied-width likelihoods, with zero/one
  latent factor and no floor/intercept floor/absorbing floor;
- width-prior mass and conditional smoothness reconstruction;
- query-outcome and realized-smoothness isolation;
- response-value parity and finite-difference gradients for all 18 supported
  saturation × carryover pairs, including zero-spend intervals;
- active-count/context/gate/regime changes with one numerical signature,
  zero controls, zero latents, reference-contribution and relative-scale priors;
- binding isolation and rejection of posterior prediction with the wrong world;
- paired full-window predictions, original input identities and unthinned stats;
- real one-shot Nutpie and two dynamic-worker fits.

The three tiny fits each used one chain, 12 tuning iterations and 8 posterior
draws (worker seed 47; one-shot seed 53). Each recorded **0 divergences**, but
these runs are far too short for convergence/ESS/calibration claims.

The existing `tests/test_oracle_sampling.py tests/test_oracle.py` regression run
also passed: **274 passed, 1 slow test skipped**. Default configuration/receipt
serialization remains unchanged. Ruff and mypy checks are separate gates.

The features-off compatibility gate also passed:

```bash
uv run --no-sync pytest tests/test_main_reproduction.py --runslow -m slow -q
```

**1 passed in 499.22 s**: generated arrays and fixed-point oracle outputs match
pinned main `ce068d5` byte for byte in the locked environment.

The broad suite exposes environment-dependent legacy MAP-pilot tests that require
a specific historical worktree and HEAD. An untouched `60122b8` checkout reproduces
**14 failures and 93 passes** in `tests/test_map_recovery_pilot.py`; this change
leaves that script and its tests unchanged. The full ordinary suite finished with
**4227 passed, 225 skipped, 14 failed** in 2035.89 s; its failure-ID set exactly
matches those 14 baseline failures, with no additional failures. These failures
are not evidence of an oracle regression and are not hidden by weakening checks.

## Read-only binding against persisted live-run samples

Used previously downloaded, integrity-validated task-00000 and task-00010 payloads
from `pymc-generator-live-dashboard`, run `structural-splits-f481c0a8fb17`.
Constructed `OracleInputs` from active observed columns, graph gates, saturation
anchors and selected context outcomes only. No fitting, validation-set selection
or source-file modification was performed.

| Preset | Weeks | Context | Channels | Controls | Width combinations | Numerical signatures |
|---|---:|---:|---:|---:|---:|---:|
| linear-none-calm | 208 | 104 | 6 | 2 | 676 | 1 |
| linear-none-hectic | 208 | 156 | 1 | 7 | 676 | 1 |

Both bindings had finite initial log densities and gradients. The first
build/bind/evaluation took 12.872 s, including lazy compilation; the second
bind/evaluation took 0.010 s. These timings are observations on this machine,
not a benchmark promise. The same worker and numerical signature were retained.

All 50 recorded presets were checked against the family inference contract:
exactly one compatible contract per each of the ten structural family pairs.
Regime-specific baseline/noise/smoothness priors are runtime data, not constants
silently inherited from the first world. Input-generation-only settings do not
split an outcome-side worker.

## Deliberate limits

No full-scale reference fits, health qualification, recovery study, SBC, remote
cost benchmark or dashboard work was performed. The consumer still owns dataset
selection, job budgets, provenance persistence, qualification and scoring.
