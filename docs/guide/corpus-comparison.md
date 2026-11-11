# Compare data with an existing corpus

Ask **which statistical properties of a query differ from those of an existing
reference corpus**. These tools do not generate or optimize a reference, run an
inference model, or assemble an interactive report.

Use [`world_descriptors` and `compare_descriptors`](world-descriptors.md) for
world-level comparisons. The additional statistics, window selection, and
standalone plots below extend that workflow. Waveform matching is a separate,
**alignment-sensitive diagnostic**, not descriptor-space nearest-world search.
Neither matching distances nor empirical score percentiles are probabilities of
origin, causal evidence, or newly calibrated hypothesis tests.

Every array on this page is freshly constructed from an independent toy example.
The assertions execute when the documentation builds; no external data, fitted
configuration, stored analysis output, or model is used.

## Select windows without selecting their values

In an application, `corpus` can be the mapping returned by `pg.load_corpus(...)`.
For this demonstration only, NumPy supplies a small, already-designed toy corpus:

```python exec="1" source="block" result="text" session="corpus-comparison"
import numpy as np
import pymc_generator as pg
from pymc_generator import viz

rng = np.random.default_rng(41)
n_worlds, horizon, length = 18, 40, 24
levels = rng.gamma(2, 1, (n_worlds, horizon, 3))
levels[rng.random(levels.shape) < 0.35] = 0
covariates = rng.normal(size=(n_worlds, horizon, 2)).cumsum(axis=1)
outcome = rng.normal(size=(n_worlds, horizon))
corpus = {
    "treatment_raw": levels,
    "covariates": covariates,
    "outcome_raw": outcome,
    "treatment_active_mask": np.ones((n_worlds, 3), dtype=bool),
    "covariate_active_mask": np.ones((n_worlds, 2), dtype=bool),
    "cell_id": np.arange(n_worlds) // 2,
}

# A caller-defined group: membership depends on metadata, never measured values.
allowed = np.ones((n_worlds, 3), dtype=bool)
allowed[:, 2] = np.arange(n_worlds) % 2 == 0
selected = pg.sample_reference_windows(
    corpus, length=length, n_windows=12, n_treatments=2, n_covariates=1,
    treatment_eligible=allowed, seed=83,
)
reference = selected.observed

# Reconstruct the returned joint worlds from the recorded lineage.
for i, (world, start) in enumerate(zip(selected.world_positions, selected.starts)):
    np.testing.assert_array_equal(
        reference.treatment_raw[i],
        levels[world, start:start + length][:, selected.treatment_indices[i]],
    )
    np.testing.assert_array_equal(
        reference.covariates[i],
        covariates[world, start:start + length][:, selected.covariate_indices[i]],
    )
    np.testing.assert_array_equal(
        reference.outcome_raw[i], outcome[world, start:start + length]
    )
print(selected.to_frame().head(3).to_string(index=False))
```

One window is selected per source world, without replacement. Every selected
series and outcome retains the **same source world and contiguous time window**.
Caller masks intersect the source active masks; requested counts may be zero.
Masks and ID overrides align with the ordered pool **after** `worlds=` selection,
before eligibility filtering. `group_ids` retains the corpus's generation cell
or `-1` when unknown; unknown does not become an independent singleton group.

For multiple caller-defined groups, supply the corresponding masks and counts in
separate calls. No column-name taxonomy or value-based role inference is applied.
An all-zero cropped window is retained, not rejected to make the corpus look more
active. The `ObservedWorlds.from_frame` rule rejecting inactive all-zero input
columns is unchanged.

!!! warning "Selection changes the reference population"
    Cropping, projection, and caller group selection define a different reference
    population. These descriptive comparisons do not establish that the original
    corpus's split-conformal typicality guarantee survives those operations.

## Compare temporal, activity, and dependence descriptors

The original descriptor selection is unchanged. New families must be selected
explicitly; signed pair lags have their own `dependence_lags` argument.

```python exec="1" source="block" result="text" session="corpus-comparison"
query_levels = rng.gamma(2, 1, (1, length, 2))
query_levels[0, :3, 0] = 0
query_levels[0, 8:13, 0] = 0
query_levels[0, -2:, 1] = 0
query = pg.ObservedWorlds(
    query_levels, rng.normal(size=(1, length, 1)), rng.normal(size=(1, length))
)
options = {
    "statistics": (
        "level_std", "difference_std", "time_correlation",
        "overlap_pearson", "diff_overlap_pearson",
        "active_spell_count", "longest_inactive_length", "lagged_pearson",
    ),
    "lags": (1, 3),
    "dependence_lags": (-2, 0, 2),
    "reducers": ("median",),
}
r = pg.world_descriptors(
    reference, source_id="synthetic-reference", world_ids=selected.world_ids,
    group_ids=selected.group_ids, **options,
)
q = pg.world_descriptors(query, source_id="synthetic-query", **options)
comparison = pg.compare_descriptors(r, q)
np.testing.assert_allclose(
    q.column("treatment_level_std_median"),
    np.median(query_levels.std(axis=1, ddof=0), axis=1),
)
feature = "treatment_level_std_median"
reference_medians = np.median(reference.treatment_raw.std(axis=1, ddof=0), axis=1)
query_median = float(np.median(query_levels.std(axis=1, ddof=0)))
np.testing.assert_allclose(
    comparison.stats(0, feature)["median_shift"], query_median - np.median(reference_medians)
)
expected_percentile = (
    np.count_nonzero(reference_medians < query_median)
    + 0.5 * np.count_nonzero(reference_medians == query_median)
) / len(reference_medians)
np.testing.assert_allclose(
    comparison.percentiles[0, comparison.features.index(feature)], expected_percentile
)
print(comparison.to_frame()[["feature", "median_shift"]].head(6).to_string(index=False))
```

| Statistic family | Definition and eligibility |
| --- | --- |
| `level_std`, `difference_std` | Population standard deviation in original units, with source-precision-aware constant detection. |
| `time_correlation` | Signed Pearson correlation with time position, a descriptive trend summary. |
| `overlap_pearson`, `diff_overlap_pearson` | Each lag's two overlapping slices are centered separately; at least three observations and resolved variation are required. |
| Activity summaries | Exact-zero inactive spells, lengths, counts, censoring, switching rate, and complete interior-run medians. |
| `lagged_pearson`, `diff_lagged_pearson` | Signed pair correlation at explicit positive, negative, or zero lags. |

The existing `acf` and `diff_acf` retain their **global-mean, full-denominator
sample ACF** definition. They are not renamed to overlap Pearson. At positive
pair lag `h`, the left series at `t` is paired with the right at `t+h`; negative
lags reverse that offset. Within-role nonzero-lag descriptors reduce **both
ordered directions**, so an arbitrary column permutation does not change them.
No lag association identifies a causal effect, confounder, or adstock kernel.

A boundary-touching spell is censored. An all-zero window is **one** doubly
boundary-censored inactive spell, not two spells. Complete-run medians remain
unavailable when no complete spell exists. `interior_inactive_fraction` is the
length of complete inactive spells divided by the **full window length**.

## Standalone diagnostic results and plots

Raw diagnostic arrays have shape `(world, time, series)`; waveform matrices in
the next section have shape `(series, time)`. When reading `ObservedWorlds`
arrays directly, pass the stored role resolution: their storage is float64 even
when their original precision was float32. `world_descriptors` handles that
metadata automatically.

```python exec="1" source="block" result="text" session="corpus-comparison"
r_temporal = pg.temporal_statistics(
    reference.treatment_raw, lags=(1, 3), resolution=reference.resolution[0]
)
q_temporal = pg.temporal_statistics(
    query.treatment_raw, lags=(1, 3), resolution=query.resolution[0]
)
r_runs = pg.activity_runs(reference.treatment_raw)
q_runs = pg.activity_runs(query.treatment_raw)
assert q_runs.values["leading_inactive_length"][0, 0] == 3
assert q_runs.values["longest_inactive_length"][0, 0] == 5

r_joint = np.concatenate(
    (reference.treatment_raw, reference.covariates, reference.outcome_raw[..., None]), axis=2
)
q_joint = np.concatenate(
    (query.treatment_raw, query.covariates, query.outcome_raw[..., None]), axis=2
)
pairs = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)]
r_precision = [reference.resolution[0]] * 2 + list(reference.resolution[1:])
q_precision = [query.resolution[0]] * 2 + list(query.resolution[1:])
r_dependence = pg.lagged_dependence(
    r_joint, pairs=pairs, lags=(-3, -1, 0, 1, 3), resolution=r_precision
)
q_dependence = pg.lagged_dependence(
    q_joint, pairs=pairs, lags=(-3, -1, 0, 1, 3), resolution=q_precision
)
print(q_runs.runs_frame().head(4).to_string(index=False))
```

Dependence plots reduce selected pairs **inside each world**, then show reference
quantiles across worlds. They never manufacture a world by mixing independently
sampled series. Reference bands are descriptive distributions, not confidence
intervals. Result tables include eligibility and observation counts; constants,
short overlaps, and missing complete spells are never replaced with zero.

## Explicit scaling and series matching

Quantile scaling is optional and must be explicit. `mode="absolute"` divides by
the chosen quantile of absolute values and preserves signs; `mode="signed"`
divides by the signed quantile and a negative divisor reverses signs and order.
Neither mode is a rank transform. With a zero divisor, original values remain
available for shape diagnostics but `scale_available` is false and normalized
MSE is unavailable. Pass the normalization result itself to `compare_series` to
retain that distinction; do not discard its metadata and mix its raw fallback
rows with normalized amplitudes.

The same normalization mode and quantile must apply to all pools. Scaling cannot
make different time alignments or incompatible measurement units meaningful.

```python exec="1" source="block" result="text" session="corpus-comparison"
cells = np.unique(selected.group_ids)
in_bank = np.isin(selected.group_ids, cells[:len(cells) // 2])
in_calibration = ~in_bank
bank_raw = reference.treatment_raw[in_bank].transpose(0, 2, 1).reshape(-1, length)
calibration_raw = reference.treatment_raw[in_calibration].transpose(0, 2, 1).reshape(-1, length)
query_raw = query.treatment_raw.transpose(0, 2, 1).reshape(-1, length)
query_scaled = pg.normalize_series_quantile(
    query_raw, quantile=0.75, mode="absolute", resolution=query.resolution[0]
)
calibration_scaled = pg.normalize_series_quantile(
    calibration_raw, quantile=0.75, mode="absolute", resolution=reference.resolution[0]
)
bank_scaled = pg.normalize_series_quantile(
    bank_raw, quantile=0.75, mode="absolute", resolution=reference.resolution[0]
)
assessment = pg.assess_series_similarity(
    query_scaled, calibration_scaled, bank_scaled, lags=(1, 3),
    query_groups=[("query", 0)] * len(query_raw),
    calibration_groups=[
        ("reference", int(cell)) for cell in np.repeat(selected.group_ids[in_calibration], 2)
    ],
    bank_groups=[("reference", int(cell)) for cell in np.repeat(selected.group_ids[in_bank], 2)],
)

# Independent brute-force MSE and empirical-rank checks, not a mock search.
for i, row in enumerate(query_scaled.values):
    errors = np.mean((bank_scaled.values - row) ** 2, axis=1)
    nearest = int(np.argmin(errors))
    assert assessment.query_matches.indices["mse"][i] == nearest
    np.testing.assert_allclose(assessment.query_matches.scores["mse"][i], errors[nearest])
    held_out = np.array([
        np.min(np.mean((bank_scaled.values - row) ** 2, axis=1))
        for row in calibration_scaled.values
    ])
    expected = (
        np.count_nonzero(held_out < errors[nearest])
        + 0.5 * np.count_nonzero(held_out == errors[nearest])
    ) / len(held_out)
    np.testing.assert_allclose(assessment.percentiles["mse"][i], expected)
print(assessment.to_frame().to_string(index=False))
```

The four metrics are pointwise `mse`, signed `level_r`, signed `diff_r`, and
`overlap_pearson_rmse`. The last compares the complete profile of the explicitly
requested overlap-Pearson lags, **not** the existing sample-ACF profile. MSE and
profile RMSE are minimized; correlations are maximized **with their sign**, not
by absolute magnitude. Undefined candidates do not win a search. No finite
candidate yields score `NaN`, index `-1`, and count `0`; ties choose the first
reference position. Nonzero MSE outside float64's representable range is
unavailable rather than a false perfect match or a silently discarded finite
mean caused by an overflowing intermediate sum.

For a single search, use `pg.compare_series(query, bank, lags=...)` followed by
`pg.nearest_series(comparison)`. The nearest result retains its producing
comparison so a waveform plot cannot silently use another candidate ordering.

Assessment uses the **same bank and metric rules** for query and held-out series.
By default, all pools must have known, pairwise-disjoint groups; repeated groups
inside a pool remain repeated. Labels share a namespace; `(source, cell)` tuples
avoid collisions between unrelated source-local IDs. Unknown groups cannot
establish disjointness. Explicitly disabling that check does not establish
independence. Finite reference counts count **series**, not independent datasets.

Percentiles are raw-score empirical CDF fractions in `[0, 1]`. A high MSE
percentile means high error; a high correlation percentile means high signed
correlation. Lower/upper bounds count scores strictly below/below-or-equal, and
the midpoint handles ties. None of these are probabilities of dataset origin.

## Render the comparisons

Each plot saves and closes a standalone figure. These executable blocks use the
site's inline-PNG pattern; no report builder or exported dataset is involved.

```python exec="1" source="block" html="1" session="corpus-comparison"
import base64
from pathlib import Path
from tempfile import TemporaryDirectory


def inline_comparison(plotter, *args, **kwargs):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "comparison.png"
        plotter(*args, str(path), **kwargs)
        image = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        '<figure class="scm-plot"><img alt="Independent synthetic comparison" '
        f'src="data:image/png;base64,{image}"></figure>'
    )


print(inline_comparison(viz.plot_temporal_comparison, r_temporal, q_temporal, series=(0,)))
```

```python exec="1" source="block" html="1" session="corpus-comparison"
print(inline_comparison(viz.plot_activity_runs, r_runs, q_runs, series=(0,)))
```

```python exec="1" source="block" html="1" session="corpus-comparison"
print(inline_comparison(viz.plot_dependence_comparison, r_dependence, q_dependence))
```

```python exec="1" source="block" html="1" session="corpus-comparison"
print(inline_comparison(
    viz.plot_series_matches, assessment.query_matches.comparison,
    assessment.query_matches, metric="mse",
))
```

```python exec="1" source="block" html="1" session="corpus-comparison"
print(inline_comparison(viz.plot_similarity_scores, assessment))
```

## Keep private inputs and outputs private

Generic tooling does not make client data or derived reports publishable.
Do not commit private inputs, statistics, curves, images, labels, source paths,
URLs, IDs, fingerprints, notebook outputs, or client-informed configurations.
Use independently generated synthetic examples for documentation and tests.
Renaming columns or removing organization names is not sufficient.
