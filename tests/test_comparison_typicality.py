"""New observable features remain usable by the existing typicality API."""

import numpy as np
import pytest

from pymc_generator import ObservedWorlds, typicality, world_descriptors
from pymc_generator.dataset_typicality import FEATURE_ROLES, TYPICALITY_STATUSES


def test_shared_activity_counts_are_not_evidence_of_copied_worlds():
    treatment = np.array(
        [
            [1, 1, 0, 0, 0, 0],
            [4, 4, 4, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 2, 0, 0, 2, 0],
            [2, 2, 2, 2, 2, 2],
            [0, 0, 0, 0, 3, 0],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
        ],
        dtype=float,
    )[..., None]
    reference = ObservedWorlds(treatment, np.ones((8, 6, 1)))
    query = ObservedWorlds(
        np.array([2, 0, 5, 0, 0, 0], dtype=float)[None, :, None], np.ones((1, 6, 1))
    )
    options = {"statistics": ("active_spell_count",), "reducers": ("median",)}
    r = world_descriptors(reference, source_id="reference", group_ids=np.arange(8), **options)
    q = world_descriptors(query, source_id="query", **options)
    feature = "treatment_active_spell_count_median"
    result = typicality(r, q, features=(feature,), k=1, seed=29)
    assert TYPICALITY_STATUSES[result.status[0]] == "ok"
    # Counts [1,1,2,2,1,1,3,3]: four below 2 and two tied at 2.
    assert result.percentile[0, 0] == pytest.approx(5 / 8)


def test_two_directed_pairs_keep_distinct_min_median_max_profiles():
    rng = np.random.default_rng(79)
    reference = ObservedWorlds(rng.normal(size=(12, 9, 2)), np.ones((12, 9, 1)))
    left = np.array([0, 2, 1, 4, 3, 7, 2, 6, 1], dtype=float)
    right = np.array([5, 0, 2, 1, 4, 3, 7, 2, 6], dtype=float)
    query = ObservedWorlds(np.stack((left, right), axis=1)[None], np.ones((1, 9, 1)))
    options = {"statistics": ("lagged_pearson",), "dependence_lags": (1,)}
    r = world_descriptors(reference, source_id="reference", group_ids=np.arange(12), **options)
    q = world_descriptors(query, source_id="query", **options)
    features = tuple(
        f"treatment_pair_lagged_pearson_lag_1_{reducer}" for reducer in ("median", "min", "max")
    )
    result = typicality(r, q, features=features, k=1, seed=31)
    forward = np.corrcoef(left[:-1], right[1:])[0, 1]
    reverse = np.corrcoef(right[:-1], left[1:])[0, 1]
    np.testing.assert_allclose(
        result.value[0],
        [np.median([forward, reverse]), min(forward, reverse), max(forward, reverse)],
    )
    assert result.n_profiled[0] == 3
    assert not np.any(result.role[0] == FEATURE_ROLES.index("duplicate"))
