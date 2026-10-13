"""Benchmark-only fixed-allocation adaptation of sampler._generate_corpus_additive.

Copied from baseline b941d3a. The scientific packing/normalization is retained.
Differences: supplied fixed structures and method adapter, one exact draw batch,
no top-up or retry, candidate-local support seed, retained-row compaction, logical
IDs in a sidecar, and coarse stage accounting. This is not production integration.
"""

from __future__ import annotations

from typing import Any

from pymc_generator.sampler import (
    _ADDITIVE_OUT_NAMES,
    _CORPUS_NATURAL_NAMES,
    _CORPUS_PARAM_NAMES,
    _CORPUS_SHOCK_NAMES,
    _CORPUS_TRAJECTORY_NAMES,
    _RETRYABLE_DRAW_ERRORS,
    CORPUS_ARRAY_FIELDS,
    CORPUS_SCHEMA_VERSION,
    CORPUS_STORAGE_MAX,
    EDGE_BASE_RATES,
    MECHANISM_ARRAY_FIELDS,
    MECHANISM_REFERENCE_FIELDS,
    PRIOR_COND_LAYOUT,
    SCHEDULE_COMPONENTS,
    TRAJECTORY_ARRAY_FIELDS,
    TRAJECTORY_COMPONENTS,
    TRAJECTORY_INPUTS,
    TRAJECTORY_PARAM_REPORTS,
    WALK_ARRAY_FIELDS,
    SCMPrior,
    _additive_task_ok,
    _finalize_corpus,
    _make_support_mask,
    _outcome_norm_overflows,
    _pack_prior_cond,
    _slice_g_active,
    _walk_sources,
    np,
    structural_key,
    time,
)


def generate_fixed(cfg: SCMPrior, adapter) -> dict[str, Any]:
    """Generate a padded additive-SCM corpus.

    The public schema is documented by :func:`sample_prior_predictive`.
    Each cell shares a discrete structure; each task draws fresh continuous
    parameters and innovations. Evaluate at active sizes, then copy accepted
    values into the preallocated maximum-size arrays. Base float32 storage
    follows float64 acceptance checks; realised rich truth remains float64.
    """
    from dataclasses import replace

    build_world_model = adapter.build

    t_start = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    layout = cfg.layout  # extended edge types
    n_query = cfg.n_query
    n_treatments_max, n_covariates_max, n_latent_max = (
        layout.n_treatments,
        layout.n_covariates,
        layout.n_latent,
    )
    n_time_steps = cfg.n_time_steps

    n_tasks = cfg.n_cells * cfg.draws_per_cell
    dimensions = {
        "task": n_tasks,
        "time": n_time_steps,
        "treatment": n_treatments_max,
        "covariate": n_covariates_max,
        "latent": n_latent_max,
        "edge": layout.n_slots,
        "shock": cfg.n_treatment_shocks,
        "indirect_source": 3,
        "component": len(TRAJECTORY_COMPONENTS),
        "treatment_jump": cfg.treatment_level_jump_count,
        "covariate_jump": cfg.covariate_level_jump_count,
    }
    corpus: dict[str, Any] = {
        key: np.zeros(tuple(dimensions[axis] for axis in axes), dtype=dtype)
        for key, (axes, dtype) in CORPUS_ARRAY_FIELDS.items()
    }
    if cfg.prior_conditioning:
        corpus["prior_cond"] = np.empty((n_tasks, len(PRIOR_COND_LAYOUT)), dtype=np.float32)
    # Optional composable-trajectory block (like prior_cond: present iff
    # configured). Schedule outputs exist in the cell models only when a schedule
    # component is enabled; otherwise activity is 1 and shifts are 0.
    trajectory_metadata = cfg.trajectory_metadata_enabled
    trajectory_schedules = cfg.trajectory_components_enabled
    if trajectory_metadata:
        for key, (axes, dtype) in TRAJECTORY_ARRAY_FIELDS.items():
            corpus[key] = np.zeros(tuple(dimensions[axis] for axis in axes), dtype=dtype)
    mechanism_fields = dict(MECHANISM_ARRAY_FIELDS) if cfg.mechanism_priors_enabled else {}
    if mechanism_fields:
        for input_type in TRAJECTORY_INPUTS:
            if getattr(cfg, f"{input_type}_reference_contribution_range") is not None:
                mechanism_fields.update(
                    {
                        key: spec
                        for key, spec in MECHANISM_REFERENCE_FIELDS.items()
                        if key.startswith(f"{input_type}_")
                    }
                )
        for key, (axes, dtype) in mechanism_fields.items():
            corpus[key] = np.zeros(tuple(dimensions[axis] for axis in axes), dtype=dtype)
    # Optional realised walk block: zero-filled, so inactive slots stay padding.
    walk_metadata = cfg.walk_parameter_metadata() if cfg.record_walk_parameters else None
    walk_fields = {
        key: spec
        for key, spec in WALK_ARRAY_FIELDS.items()
        if walk_metadata is not None and key in walk_metadata["fields"]
    }
    for key, (axes, dtype) in walk_fields.items():
        corpus[key] = np.zeros(tuple(dimensions[axis] for axis in axes), dtype=dtype)
    # Treatment normalization retains the original float64 reduction order.
    # Other accepted outputs can be cast directly into their final storage.
    treatment_raw = np.zeros((n_tasks, n_time_steps, n_treatments_max), dtype=np.float64)
    draw_fields = {
        "covariates": "covariates",
        "outcome_raw": "outcome",
        "treatment_contribution_raw": "contributions",
        "baseline_raw": "baseline",
        "latent_unobserved": "latent_unobserved",
        "indirect_effects": "indirect_effects",
        "baseline_intrinsic": "baseline_intrinsic",
        "outcome_noise": "outcome_noise",
        "covariate_contribution": "covariate_contribution",
        "latent_unobserved_contribution": "latent_unobserved_contribution",
        "indirect_effects_by_source": "indirect_effects_by_source",
        "confounding_strength": "confounding_strength",
        "treatment_shock_mask": "treatment_shock_mask",
        "treatment_shock_index": "treatment_shock_index",
        "treatment_shock_start": "treatment_shock_start",
        "treatment_shock_length": "treatment_shock_length",
        "treatment_shock_level_multiplier": "treatment_shock_level_multiplier",
        "treatment_shock_level": "treatment_shock_level",
        "treatment_level": "param_treatment_level",
        "saturation_scale": "saturation_scale",
        "carryover_alpha": "param_carryover_alpha",
        "weibull_lam": "param_weibull_lam",
        "weibull_k": "param_weibull_k",
    }
    for key in mechanism_fields:
        if key != "sat_family":
            draw_fields[key] = (
                "saturation_scale" if key == "mechanism_saturation_scale" else f"param_{key}"
            )
    n_evaluated = 0
    n_rejected = 0
    n_draw_failures = 0
    # Stratified allocation fixes every cell's treatment × covariate counts up
    # front on its own stream, seeded by one draw from ``rng``; independent
    # allocation returns None and draws the counts per cell below, exactly as
    # before.
    kept = np.zeros(n_tasks, dtype=bool)

    for cell in range(cfg.n_cells):
        adapter.current_cell = cell
        g = adapter.padded_g(cell)
        n_treatments_active, n_covariates_active, n_latent_active = adapter.counts(cell)
        rows = slice(cell * cfg.draws_per_cell, (cell + 1) * cfg.draws_per_cell)
        corpus["cell_id"][rows] = cell
        corpus["g"][rows] = layout.pack(
            g_cy=g["g_cy"],
            g_dc=g["g_dc"],
            g_dy=g["g_dy"],
            g_zy=g["g_zy"],
            g_dz=g["g_dz"],
            g_zc=g["g_zc"],
            g_cc=g["g_cc"],
            g_zz=g["g_zz"],
        )
        for key, source in (
            ("treatment_active_mask", "active_treatment"),
            ("covariate_active_mask", "active_covariate"),
            ("latent_active_mask", "active_latent"),
            ("treatment_active", "treatment_active"),
            ("n_treatments_active", "n_treatments_active"),
            ("n_covariates_active", "n_covariates_active"),
            ("n_latent_active", "n_latent_active"),
        ):
            corpus[key][rows] = g[source]
        g_act = _slice_g_active(g, n_treatments_active, n_covariates_active, n_latent_active)

        # One pm.Model per cell (fixed structure: DAG + families + smoothness);
        # the continuous params and noise are the model's RVs, drawn in batches
        # and filtered by the realism gate. Building/compiling the graph
        # dominates at large (n_treatments, n_covariates), so a model per cell
        # (not per draw) is key; FAST_COMPILE (the draw_worlds default) keeps
        # the one-off compile cheap.
        structural = adapter.cells[cell].structure
        # Prior-conditioning interval draw (per cell — one model build). False
        # returns None and consumes no RNG, preserving same-environment
        # unconditioned numerical draws, not cross-schema archive-byte identity.
        prior_cond = None
        if prior_cond is not None:
            corpus["prior_cond"][rows] = _pack_prior_cond(prior_cond)
        model, out_names, _param_names = build_world_model(
            g_act, cfg, structural, n_time_steps, prior_cond=prior_cond
        )
        trajectory_reports = (
            {
                key: report
                for key, report in TRAJECTORY_PARAM_REPORTS.items()
                if report in _param_names
            }
            if trajectory_metadata
            else {}
        )
        requested_reports = set(trajectory_reports.values()) | {
            draw_fields[key] for key in mechanism_fields if key != "sat_family"
        }
        # Leading reports join the same accepted draw without displacing the
        # legacy output traversal that assigns RNG streams.
        audit_names = (
            tuple(
                name
                for name in _param_names
                if name in requested_reports and name not in _CORPUS_PARAM_NAMES
            )
            if requested_reports
            else ()
        )
        # Recorded walk truth joins the same accepted draw ahead of the audit
        # reports. It never gates acceptance, so recording keeps every draw.
        walk_sources: dict[str, str | np.ndarray] = {}
        if walk_fields:
            active_sizes = {
                "treatment": n_treatments_active,
                "covariate": n_covariates_active,
                "latent": n_latent_active,
            }
            walk_sources = _walk_sources(cfg, walk_fields, structural, active_sizes)
        walk_names = tuple(entry for entry in walk_sources.values() if isinstance(entry, str))
        leading_names = walk_names + audit_names
        # Per-input component flags (cell structure, stored per row like the
        # carryover family). Schedule keys exist only when schedules are enabled.
        component_flags: dict[str, np.ndarray] = {}
        scheduled_treatment: np.ndarray = np.zeros(n_treatments_active, dtype=bool)
        if trajectory_metadata:
            for input_type, n_active in (
                ("treatment", n_treatments_active),
                ("covariate", n_covariates_active),
            ):
                component_flags[input_type] = np.stack(
                    [
                        np.asarray(
                            structural.get(
                                structural_key(input_type, component),
                                np.zeros(n_active, dtype=bool),
                            ),
                            dtype=np.uint8,
                        )
                        for component in TRAJECTORY_COMPONENTS
                    ],
                    axis=1,
                )
            schedule_columns = [TRAJECTORY_COMPONENTS.index(c) for c in SCHEDULE_COMPONENTS]
            scheduled_treatment = np.asarray(
                component_flags["treatment"][:, schedule_columns].any(axis=1), dtype=bool
            )
        trajectory_masks = {}
        for key in trajectory_reports:
            input_type, component_leaf = key.removeprefix("trajectory_").split("_", 1)
            component = next(
                name for name in TRAJECTORY_COMPONENTS if component_leaf.startswith(f"{name}_")
            )
            trajectory_masks[key] = (
                component_flags[input_type][:, TRAJECTORY_COMPONENTS.index(component)] != 0
            )
        accepted = 0
        cell_evaluated = 0
        cell_rejected = 0
        for _round in range(1):
            if accepted == cfg.draws_per_cell:
                break
            n_req = cfg.draws_per_cell
            try:
                # Output order covariates PyTensor's RNG traversal. Metadata
                # deterministics must precede the legacy outputs: appending
                # them changes legacy draws, while a separate same-seed call
                # produces parameters from a different joint world.
                draw_names: tuple[str, ...]
                if trajectory_schedules and cfg.n_treatment_shocks:
                    draw_names = (
                        _CORPUS_TRAJECTORY_NAMES
                        + _CORPUS_PARAM_NAMES
                        + _CORPUS_SHOCK_NAMES
                        + _ADDITIVE_OUT_NAMES
                        + _CORPUS_NATURAL_NAMES
                    )
                elif trajectory_schedules:
                    draw_names = (
                        _CORPUS_TRAJECTORY_NAMES
                        + _CORPUS_NATURAL_NAMES
                        + _CORPUS_PARAM_NAMES
                        + _CORPUS_SHOCK_NAMES
                        + _ADDITIVE_OUT_NAMES
                    )
                elif cfg.n_treatment_shocks:
                    draw_names = (
                        _CORPUS_PARAM_NAMES
                        + _CORPUS_SHOCK_NAMES
                        + _ADDITIVE_OUT_NAMES
                        + ("treatments_unshocked", "outcome_unshocked")
                    )
                else:
                    draw_names = _CORPUS_PARAM_NAMES + _CORPUS_SHOCK_NAMES + _ADDITIVE_OUT_NAMES
                rng_reference_names = draw_names
                draw_names = leading_names + draw_names
                drawn_b = adapter.draw(draw_names, n_req)
                packing_start = time.perf_counter()
                # Research extension: flags may be sampled inside the graph.
                # The adapter writes the realised flags into this same structure
                # before returning; use them for acceptance AND persisted truth.
                assert n_req == 1, "this study uses a fresh structure per world"
                if trajectory_metadata:
                    for input_type in TRAJECTORY_INPUTS:
                        component_flags[input_type][:] = np.stack(
                            [structural[structural_key(input_type, c)]
                             for c in TRAJECTORY_COMPONENTS], axis=1
                        )
                    scheduled_treatment[:] = component_flags["treatment"][:, schedule_columns].any(axis=1)
                    for key in trajectory_masks:
                        input_type, leaf = key.removeprefix("trajectory_").split("_", 1)
                        component = next(c for c in TRAJECTORY_COMPONENTS if leaf.startswith(c))
                        trajectory_masks[key] = component_flags[input_type][:, TRAJECTORY_COMPONENTS.index(component)] != 0
            except _RETRYABLE_DRAW_ERRORS:
                raise  # No retries in this fixed-allocation experiment.

            for b in range(n_req):
                if accepted == cfg.draws_per_cell:
                    break
                drawn = {name: drawn_b[name][b] for name in draw_names}
                n_evaluated += 1
                cell_evaluated += 1
                audit_finite = all(np.isfinite(drawn[name]).all() for name in audit_names)
                if not audit_finite:
                    n_rejected += 1
                    cell_rejected += 1
                    continue
                realism_arrays = (
                    {name: drawn[name] for name in rng_reference_names} if leading_names else drawn
                )

                if trajectory_schedules:
                    # Realism reads the schedule-free natural path; only a
                    # scheduled treatment's own schedule may lift its CV.
                    natural = drawn["treatments_natural"]
                    cv_alternative = None
                    if scheduled_treatment.any():
                        cv_alternative = np.where(
                            scheduled_treatment[None, :],
                            drawn["treatment_activity"]
                            * natural
                            * np.exp(drawn["treatment_log_level_shift"]),
                            natural,
                        )
                    task_ok = _additive_task_ok(
                        treatment=drawn["treatments"],
                        outcome=drawn["outcome"],
                        arrays=realism_arrays,
                        g_cy_active=g_act["g_cy"],
                        cv_floor=cfg.treatment_cv_floor,
                        realism_treatment=natural,
                        realism_outcome=drawn["outcome_natural"],
                        cv_alternative_treatment=cv_alternative,
                        # Scheduled levels multiply texture and parent terms, so
                        # extreme (validated) level priors can still overflow the
                        # float32 storage; such a draw is rejected, not stored.
                        storage_max=CORPUS_STORAGE_MAX,
                    )
                else:
                    task_ok = _additive_task_ok(
                        treatment=drawn["treatments"],
                        outcome=drawn["outcome"],
                        arrays=realism_arrays,
                        g_cy_active=g_act["g_cy"],
                        cv_floor=cfg.treatment_cv_floor,
                        realism_treatment=drawn.get("treatments_unshocked"),
                        realism_outcome=drawn.get("outcome_unshocked"),
                    )
                if not task_ok:
                    n_rejected += 1
                    cell_rejected += 1
                    continue

                support, task_split_type = _make_support_mask(
                    np.random.default_rng(np.random.SeedSequence([cfg.seed, adapter.logical_ids[cell], b, 4401])),
                    n_time_steps,
                    n_query,
                    cfg.p_long_horizon,
                )
                candidate_outcome_scale = float(np.std(drawn["outcome"][support == 1]))
                if not (np.isfinite(candidate_outcome_scale) and candidate_outcome_scale > 0.0):
                    n_rejected += 1
                    cell_rejected += 1
                    continue
                if trajectory_schedules and _outcome_norm_overflows(
                    drawn["outcome"], (n_time_steps - n_query, n_time_steps // 2)
                ):
                    # outcome_norm = outcome / scale would not be representable
                    # under either split (the validation-split repair may flip it).
                    n_rejected += 1
                    cell_rejected += 1
                    continue

                row = cell * cfg.draws_per_cell + accepted
                treatment_raw[row, :, :n_treatments_active] = drawn["treatments"]
                for key, source in draw_fields.items():
                    value = drawn[source]
                    prefix: tuple[int | slice, ...] = (
                        row,
                        *(slice(size) for size in np.shape(value)),
                    )
                    corpus[key][prefix] = value
                corpus["carryover_family"][row, :n_treatments_active] = structural[
                    "carryover_family"
                ]
                if mechanism_fields:
                    corpus["sat_family"][row, :n_treatments_active] = structural["sat_family"]
                for key, walk_source in walk_sources.items():
                    walk_value = np.ravel(
                        drawn[walk_source] if isinstance(walk_source, str) else walk_source
                    )
                    if len(walk_fields[key][0]) == 1:
                        corpus[key][row] = walk_value.item()
                    else:
                        width = walk_value.size
                        corpus[key][row, :width] = walk_value
                if trajectory_metadata:
                    corpus["treatment_components"][row, :n_treatments_active] = component_flags[
                        "treatment"
                    ]
                    corpus["covariate_components"][row, :n_covariates_active] = component_flags[
                        "covariate"
                    ]
                    for key, report in trajectory_reports.items():
                        value = np.where(trajectory_masks[key], drawn[report], 0)
                        prefix = (row, *(slice(size) for size in value.shape))
                        corpus[key][prefix] = value
                    if trajectory_schedules:
                        for key, n_active in (
                            ("treatment_activity", n_treatments_active),
                            ("covariate_activity", n_covariates_active),
                            ("treatment_log_level_shift", n_treatments_active),
                            ("covariate_level_shift", n_covariates_active),
                        ):
                            corpus[key][row, :, :n_active] = drawn[key]
                    else:
                        corpus["treatment_activity"][row, :, :n_treatments_active] = 1
                        corpus["covariate_activity"][row, :, :n_covariates_active] = 1
                corpus["support_mask"][row] = support
                corpus["is_future"][row] = task_split_type
                kept[row] = True
                adapter.accepted_ids.append((cell, b))
                accepted += 1
            adapter.times["acceptance_and_packing_s"] += time.perf_counter() - packing_start
            del drawn_b, drawn
    adapter.evaluated = n_evaluated
    adapter.rejected = n_rejected
    adapter.times["normalization_and_diagnostics_s"] = 0.0
    normalization_start = time.perf_counter()
    if not kept.any():
        return None
    corpus = {key: value[kept] for key, value in corpus.items()}
    treatment_raw = treatment_raw[kept]
    retained_ids, new_ids = np.unique(corpus["cell_id"], return_inverse=True)
    adapter.retained_cell_ids = retained_ids.tolist()
    corpus["cell_id"] = new_ids.astype(corpus["cell_id"].dtype)
    cfg = replace(cfg, n_cells=len(retained_ids))
    n_tasks = int(kept.sum())

    support_mask = corpus["support_mask"]
    split_type = corpus["is_future"]
    cell_id = corpus["cell_id"]
    treatment_active_mask = corpus["treatment_active_mask"]

    treatment_means = treatment_raw.mean(axis=1)  # (n_tasks, n_treatments)
    treatment_norm = np.divide(
        treatment_raw,
        treatment_means[:, None, :],
        out=np.zeros_like(treatment_raw),
        where=treatment_means[:, None, :] != 0.0,
    )
    active_treatment_sum = (treatment_raw * treatment_active_mask[:, None, :]).sum(
        axis=-1, keepdims=True
    )
    treatment_share = (
        np.divide(
            treatment_raw,
            active_treatment_sum,
            out=np.zeros_like(treatment_raw),
            where=active_treatment_sum != 0.0,
        )
        * treatment_active_mask[:, None, :]
    )

    corpus["treatment_raw"][:] = treatment_raw
    corpus["treatment_norm"][:] = treatment_norm
    corpus["treatment_share"][:] = treatment_share
    corpus["treatment_means"][:] = treatment_means
    del treatment_raw, treatment_norm, treatment_share, treatment_means

    # -- cell-level validation split (same logic as the legacy path) --------
    n_val_cells = max(1, int(round(cfg.val_cell_frac * cfg.n_cells)))
    if n_val_cells >= cfg.n_cells:
        n_val_cells = cfg.n_cells - 1
    val_cells = rng.permutation(cfg.n_cells)[:n_val_cells]
    corpus["is_val"][:] = np.isin(cell_id, val_cells)
    is_val = corpus["is_val"]

    val_mask = is_val == 1
    val_split_types = split_type[val_mask]
    if val_mask.sum() >= 2:
        if val_split_types.sum() == 0:
            idx_flip = np.flatnonzero(val_mask)[0]
            new_support, new_split = _make_support_mask(rng, n_time_steps, n_query, 1.0)
            support_mask[idx_flip] = new_support
            split_type[idx_flip] = new_split
        elif val_split_types.sum() == val_mask.sum():
            idx_flip = np.flatnonzero(val_mask)[0]
            new_support, new_split = _make_support_mask(rng, n_time_steps, n_query, 0.0)
            support_mask[idx_flip] = new_support
            split_type[idx_flip] = new_split

    # Derived from the FLOAT32 array that is actually persisted, not from the
    # float64 draw. `outcome_norm = outcome_raw / outcome_scale` is persisted too, so a
    # consumer must be able to reproduce both from the corpus alone; computing
    # the scale at draw precision made that impossible whenever float32 rounding
    # dominated the standard deviation. That is reachable: a short support window
    # over a very smooth walk gives a near-constant slice, where the writer and a
    # float32 recomputation disagreed by 2.5e-5 relative — past the validator's
    # 1e-5 tolerance.
    outcome_stored = corpus["outcome_raw"]
    outcome_scale = np.array(
        [
            float(np.std(outcome_stored[i][support_mask[i] == 1].astype(np.float64)))
            for i in range(n_tasks)
        ],
        dtype=np.float64,
    )
    bad_scale = ~(np.isfinite(outcome_scale) & (outcome_scale > 0.0))
    if bad_scale.any():
        full_std = np.std(outcome_stored[bad_scale].astype(np.float64), axis=1)
        outcome_scale[bad_scale] = np.where(np.isfinite(full_std) & (full_std > 0.0), full_std, 1.0)
    outcome_norm = outcome_stored.astype(np.float64) / outcome_scale[:, None]

    effective_legacy_edge_rates = {**EDGE_BASE_RATES, **(cfg.edge_rate_overrides or {})}

    # -- static diagnostics --------------------------------------------------
    # Every entry here is a deterministic function of (config, seed) EXCEPT the
    # "timing" sub-block, which quarantines the wall clock so ``save_corpus``
    # can drop it and persist byte-identical shards for a same-seed rerun.
    diagnostics = {
        "edge_types": list(layout.edge_types),
        "draws_per_cell": int(cfg.draws_per_cell),
        "short_horizon_n_query": int(n_query),
        "n_draws_evaluated": int(n_evaluated),
        "rejection_rate": float(n_rejected / max(n_evaluated, 1)),
        # Draw batches that raised a retryable numeric failure and were
        # resampled. These produced no candidates, so they are deliberately
        # absent from n_draws_evaluated / rejection_rate: a nonzero value here
        # means the GRAPH misbehaved, not that the realism filter is strict.
        "n_draw_failures": int(n_draw_failures),
        "edge_base_rates": {
            "cy": float(effective_legacy_edge_rates["cy"]),
            "dc": float(effective_legacy_edge_rates["dc"]),
            "dz": float(cfg.dz_base_rate),
            "dy": float(effective_legacy_edge_rates["dy"]),
            "zy": float(effective_legacy_edge_rates["zy"]),
            "zc": float(cfg.zc_base_rate),
            "cc": float(cfg.cc_base_rate),
            "zz": float(cfg.zz_base_rate),
        },
        "edge_budget": (
            {
                et: list(v) if isinstance(v, (tuple, list)) else int(v)
                for et, v in cfg.edge_budget.items()
            }
            if cfg.edge_budget
            else None
        ),
        "min_no_direct_effect_treatments": int(cfg.min_no_direct_effect_treatments),
        "schema_version": CORPUS_SCHEMA_VERSION,
        # Wall clock lives under its own key so the rest of the block stays a
        # pure function of (config, seed); _finalize_corpus adds tasks_per_sec
        # next to it and save_corpus drops the whole block.
        "timing": {"elapsed_s": float(time.perf_counter() - t_start)},
    }
    if cfg.mechanism_priors_enabled:
        diagnostics["mechanism_priors"] = {
            "parameter_fields": list(mechanism_fields),
            "saturation_prior_ranges": {
                family: {name: [float(lo), float(hi)] for name, (lo, hi) in parameters.items()}
                for family, parameters in cfg.saturation_prior_ranges.items()
            },
            "mm_scale_prior": cfg.mm_scale_prior,
            "treatment_reference_contribution_range": (
                [float(x) for x in cfg.treatment_reference_contribution_range]
                if cfg.treatment_reference_contribution_range is not None
                else None
            ),
            "treatment_reference_multiplier": float(cfg.treatment_reference_multiplier),
            "covariate_reference_contribution_range": (
                [float(x) for x in cfg.covariate_reference_contribution_range]
                if cfg.covariate_reference_contribution_range is not None
                else None
            ),
            "covariate_reference_scale": float(cfg.covariate_reference_scale),
        }
    if cfg.prior_conditioning:
        # Self-describing .npz (as with the signal block): echo the layout,
        # the supports, and the width ranges so consumers derive feature
        # scaling from the corpus instead of duplicating constants.
        spec = cfg.prior_cond_spec()
        diagnostics["prior_cond"] = {
            "layout": list(PRIOR_COND_LAYOUT),
            "supports": {q: list(s["support"]) for q, s in spec.items()},
            "width_ranges": {q: list(s["width_range"]) for q, s in spec.items()},
        }
    if walk_metadata is not None:
        diagnostics["walk_parameters"] = walk_metadata

    corpus["outcome_norm"][:] = outcome_norm
    corpus["outcome_scale"][:] = outcome_scale
    corpus["diagnostics"] = diagnostics
    corpus = _finalize_corpus(corpus, cfg)
    adapter.times["normalization_and_diagnostics_s"] = time.perf_counter() - normalization_start
    return corpus
