# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "arviz==1.3.0",
#     "h5netcdf==1.8.1",
#     "h5py==3.16.0",
#     "marimo==0.25.1",
#     "matplotlib==3.11.2",
#     "numpy==2.4.6",
#     "nutpie==0.16.11",
#     "pandas==3.0.6",
#     "plotly==7.1.0",
#     "pymc==6.3.2",
#     "pymc-extras==0.15.1",
#     "pymc-generator",
#     "pytensor==3.3.3",
#     "scikit-learn==1.9.1",
#     "scipy==1.18.1",
#     "threadpoolctl==3.7.0",
#     "xarray==2026.9.0",
# ]
#
# [tool.uv.sources]
# pymc-generator = { path = "../", editable = true }
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from plotly.colors import qualitative
    from textwrap import wrap
    from pymc_generator.slots import TRAJECTORY_COMPONENTS
    import numpy as np
    import pymc_generator as pg

    return (
        TRAJECTORY_COMPONENTS,
        go,
        make_subplots,
        mo,
        np,
        pg,
        qualitative,
        wrap,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # OLS reporting

    An interactive workspace for exploring generated-dataset identifiability.
    """)
    return


@app.cell
def showcase_configuration(StudyPath, mo, pg, study_json):
    # Persist the original showcase recipe explicitly; historical results use it.
    GENERATION_SEED = 29
    showcase_cfg = pg.SCMPrior(**study_json.loads(
        StudyPath("notebooks/interactive_batches_legacy/config_29.json").read_text()
    ))

    mo.md("""
    ## Generate two constant-baseline cells at a time

    Each batch contains **two cells, one world per cell**, using the showcase's
    media/control mechanisms, noise, 104-week horizon and weighted active counts.
    For future batches the nonzero-effect counts are uniform: **1–N media and 0–M controls**, conditional on the active input counts. The current 50-world study has been regenerated with this recipe.
    Intrinsic baseline variation is disabled (`rw_baseline_std_range=(0, 0)`).
    The constant baseline level still varies between worlds and will be estimated
    as an OLS intercept. Control contributions remain time-varying.

    **Generate 2 more cells** appends a batch with a new seed without regenerating
    previous batches. Stratification applies within each batch, not globally.
    The dropdown shows constant-baseline worlds only. Earlier varying-baseline
    batches are retained in `batches` for later study, unchanged.

    Batches are held in memory, not saved to disk. Restarting the kernel or
    rerunning the state-initialization cell resets this collection.
    """)
    return (showcase_cfg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate two constant-baseline cells at a time

    Each batch contains **two cells, one world per cell**, using the showcase's
    media/control mechanisms, noise, 104-week horizon and weighted active counts.
    Only intrinsic baseline variation is disabled (`rw_baseline_std_range=(0, 0)`).
    The constant baseline level still varies between worlds and will be estimated
    as an OLS intercept. Control contributions remain time-varying.

    **Generate 2 more cells** appends a batch with a new seed without regenerating
    previous batches. Stratification applies within each batch, not globally.
    The dropdown shows constant-baseline worlds only. Earlier varying-baseline
    batches are retained in `batches` for later study, unchanged.

    Batches are held in memory, not saved to disk. Restarting the kernel or
    rerunning the state-initialization cell resets this collection.
    """)
    return


@app.cell
def batch_controls(StudyPath, constant_baseline_cfg, mo, pg, replace):
    constant_baseline_cfg.validate()
    _initial_batches=tuple({"seed":10029+batch,
        "config":replace(constant_baseline_cfg,seed=10029+batch),
        "corpus":pg.load_corpus(StudyPath(f"notebooks/ols_study_50_seed10029/batch_{batch:02d}.npz")),
    } for batch in range(25))
    get_batches,set_batches=mo.state(_initial_batches)
    generate_more=mo.ui.button(value=len(_initial_batches)-1,on_click=lambda batch_index:batch_index+1,label="Generate 2 more cells")
    generate_more
    return generate_more, get_batches, set_batches


@app.cell
def constant_baseline_configuration(replace, showcase_cfg):
    # Current generation recipe: constant baseline and uniform nonzero-effect counts.
    constant_baseline_cfg = replace(showcase_cfg, rw_baseline_std_range=(0.0, 0.0),
        edge_budget={**showcase_cfg.edge_budget,"cy":(1,showcase_cfg.n_treatments),"zy":(0,showcase_cfg.n_covariates)})
    assert constant_baseline_cfg.outcome_std_mode == "relative"
    constant_baseline_cfg.validate()
    return (constant_baseline_cfg,)


@app.cell
def uniform_effect_count_configuration(constant_baseline_cfg, mo):
    uniform_effect_count_cfg = constant_baseline_cfg
    mo.md("**Current generation:** uniformly sample 1–N contributing media and 0–M contributing controls, "
          "conditional on N/M inputs present; choose identities uniformly. All study analysis uses this modified recipe.")
    return (uniform_effect_count_cfg,)


@app.cell
def generate_batch(
    generate_more,
    get_batches,
    mo,
    np,
    pg,
    replace,
    set_batches,
    uniform_effect_count_cfg,
):
    batch_index = generate_more.value
    _existing = get_batches()
    if batch_index == len(_existing):
        _batch_cfg = replace(uniform_effect_count_cfg, seed=10029 + batch_index)
        with mo.status.spinner(title=f"Generating two cells (seed {_batch_cfg.seed})…"):
            _corpus = pg.sample_prior_predictive(_batch_cfg)
        _errors = pg.DataGenerator.validate_corpus(_corpus)
        if _errors:
            raise ValueError(f"Generated corpus failed validation: {_errors}")
        assert len(_corpus["cell_id"]) == 2
        assert len(np.unique(_corpus["cell_id"])) == 2
        assert np.all(np.ptp(_corpus["baseline_intrinsic"], axis=1) == 0)
        set_batches(_existing + ({"seed": _batch_cfg.seed, "config": _batch_cfg, "corpus": _corpus},))
    elif batch_index > len(_existing):
        raise RuntimeError("A batch failed. Rerun this cell before requesting another batch.")
    return


@app.cell
def world_selector(get_batches, mo):
    batches = get_batches()
    world_options = {
        f"World {2 * batch_id + world_id + 1} · batch {batch_id} · seed {batch['seed']}": (batch_id, world_id)
        for batch_id, batch in enumerate(batches)
        if batch["config"].rw_baseline_std_range == (0.0, 0.0)
        for world_id in range(len(batch["corpus"]["cell_id"]))
    }
    world_picker = mo.ui.dropdown(
        options=world_options,
        value=next(iter(world_options)),
        label="View constant-baseline world",
    )
    world_picker
    return batches, world_picker


@app.cell(hide_code=True)
def input_plot_function(
    TRAJECTORY_COMPONENTS,
    go,
    make_subplots,
    np,
    qualitative,
    wrap,
):
    def plot_world_inputs(corpus, world_id, *, title):
        labels = {
            "hf": "high-frequency noise", "pulse": "pulses",
            "onset": "onset", "offset": "shutdown", "flighting": "flighting",
            "level_jump": "level jumps", "seasonal": "seasonality", "trend": "trend",
        }
        panels = (
            ("treatment", "treatment_raw", "C", "Media (before adstock / saturation)"),
            ("covariate", "covariates", "Z", "Controls"),
        )
        titles = []
        for role, key, prefix, panel in panels:
            slots = np.flatnonzero(corpus[f"{role}_active_mask"][world_id])
            flags = corpus[f"{role}_components"][world_id, slots].any(axis=0)
            patterns = ", ".join(
                labels.get(component, component.replace("_", " "))
                for component, included in zip(TRAJECTORY_COMPONENTS, flags, strict=True)
                if included
            ) or "none"
            titles.append(panel + "<br><sup>" + "<br>".join(
                wrap("Sampled component flags: " + patterns, width=100)
            ) + "</sup>")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.23, subplot_titles=titles)
        for row, (role, key, prefix, panel) in enumerate(panels, start=1):
            slots = np.flatnonzero(corpus[f"{role}_active_mask"][world_id])
            for slot in slots:
                name = f"{prefix}{slot + 1}"
                fig.add_trace(go.Scatter(
                    x=np.arange(corpus[key].shape[1]),
                    y=corpus[key][world_id, :, slot],
                    mode="lines", name=name,
                    legend="legend" if row == 1 else "legend2",
                    line={"color": qualitative.Dark24[int(slot) % len(qualitative.Dark24)], "width": 1.5},
                    hovertemplate=name + "<br>Week %{x}<br>Input %{y:.4g}<extra></extra>",
                ), row=row, col=1)
            fig.update_yaxes(title_text="Input units", zeroline=True, row=row, col=1)
        media_count = int(corpus["treatment_active_mask"][world_id].sum())
        control_count = int(corpus["covariate_active_mask"][world_id].sum())
        fig.update_layout(
            title={"text": f"{title} — {media_count} active media · {control_count} active controls",
                   "font": {"size": 17}},
            template="plotly_white", height=850,
            margin={"t": 145, "b": 145, "l": 65, "r": 25},
            legend={"orientation": "h", "x": 0, "xanchor": "left",
                    "y": 0.55, "yanchor": "top",
                    "itemclick": "toggle", "itemdoubleclick": "toggleothers"},
        )
        fig.update_layout(legend2={
            "orientation": "h", "x": 0, "xanchor": "left",
            "y": -0.14, "yanchor": "top",
            "itemclick": "toggle", "itemdoubleclick": "toggleothers",
        })
        fig.update_annotations(font_size=12)
        fig.update_xaxes(title_text="Reported week (zero-based)", row=2, col=1)
        return fig

    return (plot_world_inputs,)


@app.cell(hide_code=True)
def world_input_plot(batches, mo, plot_world_inputs, world_picker):
    selected_batch, selected_world = world_picker.value
    input_figure = plot_world_inputs(
        batches[selected_batch]["corpus"],
        selected_world,
        title=f"World {2 * selected_batch + selected_world + 1} · seed {batches[selected_batch]['seed']}",
    )
    mo.ui.plotly(input_figure)
    return selected_batch, selected_world


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compiled media transformation

    `media_transform(media, alpha, half_saturation)` maps one nonnegative time
    series to normalized geometric adstock followed by unit-asymptote
    Michaelis–Menten saturation. Inputs and output use float64.

    The kernel has `l_max` lags and zero prehistory. `alpha` is in [0, 1];
    `half_saturation` is positive, in media-input units. For generator truth,
    `half_saturation = mm_kappa_mult * mechanism_saturation_scale`.
    Only this product is identifiable from the response curve. Linear coefficients
    are left for OLS, not included in the transform.

    `transform_media_columns` applies the same compiled function to a
    (time, channel) matrix, with one decay and half-saturation value per channel.
    It does not recompile when parameters, inputs or the selected world change.

    **Prehistory:** generation used hidden burn-in inputs. The returned array keeps
    all reported weeks, but its first `l_max - 1` rows need not match generator
    truth. Checks below compare the remaining rows; the fitting-window choice is
    still open. Supply full history and slice the output if that history becomes
    available. No OLS fitting is performed here.
    """)
    return


@app.cell
def transform_imports():
    import pytensor
    import pytensor.tensor as pt
    from pytensor.raise_op import Assert
    from pymc_generator import mechanisms
    from pymc_generator.slots import SlotLayout

    return Assert, SlotLayout, mechanisms, pt, pytensor


@app.cell
def compiled_media_transform(
    Assert,
    mechanisms,
    np,
    pt,
    pytensor,
    showcase_cfg,
):
    def compile_media_transform(l_max):
        if isinstance(l_max, bool) or not isinstance(l_max, (int, np.integer)) or l_max < 1:
            raise ValueError("l_max must be a positive integer")
        media = pt.dvector("media")
        alpha = pt.dscalar("alpha")
        half_saturation = pt.dscalar("half_saturation")
        checked_media = Assert("media must be finite, nonnegative and nonempty")(
            media, media.size > 0, pt.all(media >= 0), pt.all(~pt.isinf(media))
        )
        checked_alpha = Assert("alpha must lie in [0, 1]")(alpha, alpha >= 0, alpha <= 1)
        checked_half = Assert("half_saturation must be finite and positive")(
            half_saturation, half_saturation > 0, ~pt.isinf(half_saturation)
        )
        adstocked = mechanisms.apply_geometric_carryover(
            checked_media[:, None], checked_alpha, l_max
        )[:, 0]
        saturated = mechanisms.stable_michaelis_menten_kappa_relative(
            adstocked, 1.0, kappa_mult=checked_half
        )
        return pytensor.function(
            [media, alpha, half_saturation], saturated,
            name="geometric_michaelis_menten_media", mode="FAST_RUN",
        )

    media_transform = compile_media_transform(showcase_cfg.l_max)


    def transform_media_columns(media, alpha, half_saturation):
        media = np.asarray(media, dtype=np.float64)
        alpha = np.asarray(alpha, dtype=np.float64)
        half_saturation = np.asarray(half_saturation, dtype=np.float64)
        if media.ndim != 2 or media.shape[0] == 0 or media.shape[1] == 0:
            raise ValueError("media must have nonempty (time, channel) shape")
        if alpha.shape != (media.shape[1],) or half_saturation.shape != alpha.shape:
            raise ValueError("one alpha and half_saturation value required per channel")
        return np.column_stack([
            media_transform(media[:, channel], alpha[channel], half_saturation[channel])
            for channel in range(media.shape[1])
        ])

    return compile_media_transform, media_transform, transform_media_columns


@app.cell
def selected_media_transform(
    batches,
    mo,
    np,
    selected_batch,
    selected_world,
    transform_media_columns,
):
    selected_corpus = batches[selected_batch]["corpus"]
    active_media_slots = np.flatnonzero(selected_corpus["treatment_active_mask"][selected_world])
    media_inputs = selected_corpus["treatment_raw"][selected_world][:, active_media_slots].astype(np.float64)
    true_alpha = selected_corpus["carryover_alpha"][selected_world, active_media_slots].astype(np.float64)
    true_half_saturation = (
        selected_corpus["mm_kappa_mult"][selected_world, active_media_slots]
        * selected_corpus["mechanism_saturation_scale"][selected_world, active_media_slots]
    )
    transformed_media = transform_media_columns(media_inputs, true_alpha, true_half_saturation)
    mo.md(f"Compiled transform ready: **{transformed_media.shape[0]} weeks × {transformed_media.shape[1]} active media**. No linear coefficients applied.")
    return (selected_corpus,)


@app.cell
def transform_checks(
    SlotLayout,
    batches,
    media_transform,
    mo,
    np,
    showcase_cfg,
    transform_media_columns,
):
    def check_media_transform(available_batches):
        # Independent finite-convolution reference, including boundary decays.
        for decay in (0.0, 0.4, 1.0):
            for values in (np.zeros(12), np.eye(1, 12)[0], np.arange(12, dtype=float)):
                kernel = decay ** np.arange(showcase_cfg.l_max)
                kernel /= kernel.sum()
                carried = np.convolve(values, kernel)[:len(values)]
                np.testing.assert_allclose(
                    media_transform(values, decay, 2.0), carried / (2.0 + carried),
                    rtol=1e-12, atol=1e-14,
                )
        for values, decay, half in (
            ([-1.0], 0.5, 1.0), ([np.nan], 0.5, 1.0), ([np.inf], 0.5, 1.0),
            ([1.0], -0.1, 1.0), ([1.0], 1.1, 1.0), ([1.0], np.nan, 1.0),
            ([1.0], 0.5, 0.0), ([1.0], 0.5, np.inf), ([1.0], 0.5, np.nan),
        ):
            try:
                media_transform(np.asarray(values), decay, half)
            except AssertionError:
                pass
            else:
                raise AssertionError("Invalid transform input was accepted")

        max_error = 0.0
        world_count = 0
        for batch in available_batches:
            corpus, cfg = batch["corpus"], batch["config"]
            assert cfg.l_max == showcase_cfg.l_max
            layout = SlotLayout(cfg.n_treatments, cfg.n_covariates, cfg.n_latent)
            start = cfg.l_max - 1 if cfg.carryover_burn_in else 0
            for world in range(len(corpus["cell_id"])):
                slots = np.flatnonzero(corpus["treatment_active_mask"][world])
                assert (corpus["carryover_family"][world, slots] == 1).all()
                assert (corpus["sat_family"][world, slots] == 3).all()
                response = transform_media_columns(
                    corpus["treatment_raw"][world][:, slots],
                    corpus["carryover_alpha"][world, slots],
                    corpus["mm_kappa_mult"][world, slots]
                    * corpus["mechanism_saturation_scale"][world, slots],
                )
                direct_edges = layout.unpack(corpus["g"][world])["cy"][slots]
                reconstructed = response * corpus["beta"][world, slots] * direct_edges
                truth = corpus["treatment_contribution_raw"][world][:, slots]
                # Stored inputs/contributions are float32; calculations are float64.
                np.testing.assert_allclose(reconstructed[start:], truth[start:], rtol=3e-6, atol=3e-6)
                max_error = max(max_error, float(np.max(np.abs(reconstructed[start:] - truth[start:]))))
                world_count += 1
        return world_count, max_error

    checked_worlds, transform_max_error = check_media_transform(batches)
    mo.md(
        f"**Transform checks passed:** independent NumPy convolution, invalid-input rejection, "
        f"and {checked_worlds} loaded worlds. Maximum contribution error after the "
        f"prehistory-dependent rows: **{transform_max_error:.2g}**. "
        "This verifies the transform at true parameters, not OLS recovery."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Four OLS recovery fits

    We cross **estimated / true media transformations** with **observed / noiseless
    outcomes**. “Noiseless” means subtracting the stored observation-noise realization,
    not generating a different world. The true graph now selects nonzero-effect media and controls for fitting; known-null contributions are fixed to zero. All active inputs remain in the reporting layout.
    The intercept estimates the constant intrinsic baseline. Coefficients are
    unconstrained (ordinary OLS, not nonnegative least squares).

    Estimated-transform fits use variable projection: for each candidate decay and
    half-saturation, reuse the compiled transform and solve the linear coefficients
    by `numpy.linalg.lstsq`. SciPy then minimizes the residual vector. This is a
    **local nonlinear least-squares search**, not a guaranteed global solution.
    Two deterministic, non-truth-initialized starts use at most 120 solver evaluations
    each; finite-difference residual calls are counted separately. Decays are bounded
    in [0, 1]. Half-saturation is optimized in log space between 0.001 and 1000 times
    each channel's mean positive observed input (numerical search bounds, not priors).

    **Common window:** transform all 104 inputs, then fit and score weeks 7–103
    (97 observations), excluding dependence on unavailable burn-in history. Neither
    OLS nor recovery scoring uses the first seven rows. Results are in-sample.

    Outcome fit and component recovery are reported separately. Component relative
    L2 error is `norm(estimated - true) / norm(true)`, over all time/term entries;
    zero truth has undefined relative error. A small outcome residual alone does
    not establish identifiable contributions. Null-channel transformation parameters
    are not identifiable. No “good recovery” threshold has been chosen.
    """)
    return


@app.cell
def ols_imports():
    from scipy.optimize import least_squares
    from time import perf_counter

    return least_squares, perf_counter


@app.cell
def ols_fitting_functions(
    SlotLayout,
    least_squares,
    np,
    perf_counter,
    showcase_cfg,
    transform_media_columns,
):
    def solve_linear_ols(design, target):
        # Column scaling improves conditioning without changing the OLS problem.
        scale = np.sqrt(np.mean(design ** 2, axis=0))
        scale = np.where(scale > 0, scale, 1.0)
        solution, _, rank, singular = np.linalg.lstsq(design / scale, target, rcond=None)
        coefficient = solution / scale
        condition = float(singular[0] / singular[-1]) if rank == design.shape[1] else float("inf")
        return coefficient, design @ coefficient, int(rank), condition


    def fit_profiled_ols(media, controls, target, start, *, fixed=None, starts=2, max_nfev=120):
        began = perf_counter()
        n_media = media.shape[1]
        calls = 0

        def solve(alpha, half):
            transformed = transform_media_columns(media, alpha, half)
            design = np.column_stack([np.ones(len(media)), transformed, controls])[start:]
            coef, prediction, rank, condition = solve_linear_ols(design, target[start:])
            return design, coef, prediction, rank, condition

        attempts = []
        if fixed is not None:
            alpha, half = fixed
            status = "linear solve"
            success = True
            evaluations = 0
            boundary_count = 0
        else:
            input_scale = np.array([
                np.mean(column[column > 0]) if np.any(column > 0) else 1.0
                for column in media.T
            ])
            lower = np.r_[np.zeros(n_media), np.full(n_media, np.log(0.001))]
            upper = np.r_[np.ones(n_media), np.full(n_media, np.log(1000.0))]
            target_scale = max(float(np.std(target[start:])), 1e-8)

            def residual(theta):
                nonlocal calls
                calls += 1
                _, _, prediction, _, _ = solve(theta[:n_media], input_scale * np.exp(theta[n_media:]))
                return (prediction - target[start:]) / target_scale

            rng = np.random.default_rng(2026)
            candidates = []
            for attempt in range(starts):
                initial = (np.r_[np.full(n_media, 0.5), np.zeros(n_media)] if attempt == 0 else
                           np.r_[rng.uniform(0.15, 0.85, n_media), rng.uniform(-1, 1, n_media)])
                result = least_squares(residual, initial, bounds=(lower, upper),
                                       max_nfev=max_nfev, ftol=1e-8, xtol=1e-8, gtol=1e-8)
                candidates.append(result)
                attempts.append({"start": attempt + 1, "success": bool(result.success),
                                 "nfev": int(result.nfev), "cost": float(result.cost),
                                 "message": str(result.message)})
            best = min(candidates, key=lambda result: result.cost)
            alpha = best.x[:n_media]
            half = input_scale * np.exp(best.x[n_media:])
            success = bool(best.success)
            status = str(best.message)
            evaluations = sum(item["nfev"] for item in attempts)
            boundary_count = int(np.count_nonzero(best.active_mask))
        design, coef, prediction, rank, condition = solve(alpha, half)
        return {"alpha": np.asarray(alpha).copy(), "half": np.asarray(half).copy(),
                "coefficients": coef, "prediction": prediction, "components": design * coef,
                "rank": rank, "columns": design.shape[1], "condition": condition,
                "success": success, "message": status, "solver_nfev": evaluations,
                "residual_calls": calls, "boundary_count": boundary_count,
                "attempts": attempts, "seconds": perf_counter() - began}


    def prepare_ols_case(corpus, world, cfg):
        if cfg.l_max != showcase_cfg.l_max:
            raise ValueError("The compiled transform uses a different lag length")
        baseline = corpus["baseline_intrinsic"][world].astype(np.float64)
        if np.ptp(baseline) != 0:
            raise ValueError("This OLS model requires a constant intrinsic baseline")
        if np.any(corpus["latent_unobserved_contribution"][world]) or np.any(corpus["indirect_effects_by_source"][world]):
            raise ValueError("This initial OLS model excludes latent and indirect contributions")
        media_slots = np.flatnonzero(corpus["treatment_active_mask"][world])
        control_slots = np.flatnonzero(corpus["covariate_active_mask"][world])
        if not np.all(corpus["carryover_family"][world, media_slots] == 1) or not np.all(corpus["sat_family"][world, media_slots] == 3):
            raise ValueError("Expected geometric adstock and Michaelis–Menten saturation")
        graph = SlotLayout(cfg.n_treatments, cfg.n_covariates, cfg.n_latent).unpack(corpus["g"][world])
        media = corpus["treatment_raw"][world][:, media_slots].astype(np.float64)
        controls = corpus["covariates"][world][:, control_slots].astype(np.float64)
        observed = corpus["outcome_raw"][world].astype(np.float64)
        noiseless = observed - corpus["outcome_noise"][world].astype(np.float64)
        start = cfg.l_max - 1 if cfg.carryover_burn_in else 0
        truth = np.column_stack([
            baseline, corpus["treatment_contribution_raw"][world][:, media_slots],
            corpus["covariate_contribution"][world][:, control_slots],
        ])[start:].astype(np.float64)
        np.testing.assert_allclose(truth.sum(axis=1), noiseless[start:], rtol=3e-6, atol=3e-6)
        return {"media_effect_mask": graph["cy"][media_slots].astype(bool), "control_effect_mask": graph["zy"][control_slots].astype(bool), "media": media, "controls": controls, "observed": observed,
                "noiseless": noiseless, "start": start, "truth": truth,
                "names": ["Intercept"] + [f"C{i + 1}" for i in media_slots] + [f"Z{i + 1}" for i in control_slots],
                "true_coefficients": np.r_[baseline[0], corpus["beta"][world, media_slots] * graph["cy"][media_slots],
                                            corpus["rho_zy"][world, control_slots] * graph["zy"][control_slots]],
                "alpha": corpus["carryover_alpha"][world, media_slots].astype(np.float64),
                "half": (corpus["mm_kappa_mult"][world, media_slots] * corpus["mechanism_saturation_scale"][world, media_slots]).astype(np.float64)}


    def relative_component_error(estimate, truth):
        denominator = np.linalg.norm(truth)
        return float(np.linalg.norm(estimate - truth) / denominator) if denominator > 0 else float("nan")


    def run_four_ols_fits(case):
        results = []
        for label, estimate, noisy in (
            ("1 · Estimated transforms + noise", True, True),
            ("2 · Estimated transforms, no noise", True, False),
            ("3 · True transforms + noise", False, True),
            ("4 · True transforms, no noise", False, False),
        ):
            target = case["observed"] if noisy else case["noiseless"]
            fit = fit_oracle_graph_ols(case, target, fixed=not estimate)
            fit.update(label=label, estimated_transforms=estimate, noisy=noisy,
                       target=target[case["start"]:])
            results.append(fit)
        return results

    def fit_oracle_graph_ols(case, target, *, fixed=False, solver=None):
        media_mask=case["media_effect_mask"]
        control_mask=case["control_effect_mask"]
        media=case["media"][:,media_mask]
        controls=case["controls"][:,control_mask]
        if solver is None:
            fit=fit_profiled_ols(media,controls,target,case["start"],
                fixed=(case["alpha"][media_mask],case["half"][media_mask]) if fixed else None)
        else:
            fit=solver(media,controls,target,case["start"])
        n_media=len(media_mask);n_controls=len(control_mask)
        positions=np.r_[0,1+np.flatnonzero(media_mask),1+n_media+np.flatnonzero(control_mask)]
        coefficients=np.zeros(1+n_media+n_controls)
        coefficients[positions]=fit["coefficients"]
        components=np.zeros((len(fit["prediction"]),len(coefficients)))
        components[:,positions]=fit["components"]
        # Placeholders only for known-null shapes: never fitted, never interpreted as estimates.
        alpha=np.full(n_media,0.5)
        half=np.maximum(np.max(case["media"],axis=0),1.0)
        alpha[media_mask]=fit["alpha"];half[media_mask]=fit["half"]
        if "constraint_fraction" in fit:
            fractions=np.ones(n_media)
            fractions[media_mask]=fit["constraint_fraction"]
            fit["constraint_fraction"]=fractions
        fit.update(coefficients=coefficients,components=components,alpha=alpha,half=half,oracle_graph=True)
        np.testing.assert_allclose(components.sum(axis=1),fit["prediction"],rtol=1e-12,atol=1e-12)
        assert np.all(components[:,1:1+n_media][:,~media_mask]==0)
        assert np.all(components[:,1+n_media:][:,~control_mask]==0)
        return fit


    return (
        fit_oracle_graph_ols,
        fit_profiled_ols,
        prepare_ols_case,
        relative_component_error,
        run_four_ols_fits,
        solve_linear_ols,
    )


@app.cell
def ols_run_button(mo):
    run_ols = mo.ui.run_button(label="Fit all four OLS variants for the selected world")
    run_ols
    return (run_ols,)


@app.cell
def fit_selected_world(
    batches,
    mo,
    prepare_ols_case,
    run_four_ols_fits,
    run_ols,
    selected_batch,
    selected_corpus,
    selected_world,
):
    mo.stop(not run_ols.value, mo.md("Press **Fit all four OLS variants** for this world. Changing worlds requires another click."))
    ols_case = prepare_ols_case(selected_corpus, selected_world, batches[selected_batch]["config"])
    with mo.status.spinner(title="Fitting four OLS variants for the selected world…"):
        fit_results = run_four_ols_fits(ols_case)
    fit_world_label = f"World {2 * selected_batch + selected_world + 1} · seed {batches[selected_batch]['seed']}"
    return fit_results, fit_world_label, ols_case


@app.cell
def ols_fit_summary(
    fit_results,
    fit_world_label,
    mo,
    np,
    ols_case,
    relative_component_error,
):
    fit_summary = [
        {
            "Fit": result["label"],
            "Target RMSE": float(np.sqrt(np.mean((result["prediction"] - result["target"]) ** 2))),
            "Noiseless outcome RMSE": float(np.sqrt(np.mean((result["prediction"] - ols_case["noiseless"][ols_case["start"]:]) ** 2))),
            "Media contribution relative L2": relative_component_error(result["components"][:, 1:1 + ols_case["media"].shape[1]], ols_case["truth"][:, 1:1 + ols_case["media"].shape[1]]),
            "Control contribution relative L2": relative_component_error(result["components"][:, 1 + ols_case["media"].shape[1]:], ols_case["truth"][:, 1 + ols_case["media"].shape[1]:]),
            "Intercept error": float(result["coefficients"][0] - ols_case["true_coefficients"][0]),
            "Rank / columns": f"{result['rank']} / {result['columns']}",
            "Scaled condition number": result["condition"],
            "Solver success": result["success"],
            "At bounds": result["boundary_count"],
            "Residual calls": result["residual_calls"],
            "Seconds": result["seconds"],
            "Status": result["message"],
        }
        for result in fit_results
    ]
    mo.vstack([mo.md(f"### {fit_world_label} — four fits, {len(ols_case['truth'])} fitted weeks"),
               mo.ui.table(fit_summary, selection=None),
               mo.md("**Caution:** " + "; ".join(
                   result["label"] + ": " + result["message"]
                   for result in fit_results if not result["success"]
               )) if any(not result["success"] for result in fit_results)
               else mo.md("Solver termination is not a contribution-recovery certificate.")])
    return


@app.cell
def ols_fit_picker(fit_results, mo):
    fit_picker = mo.ui.dropdown(
        options={result["label"]: index for index, result in enumerate(fit_results)},
        value=fit_results[0]["label"], label="Inspect fit",
    )
    fit_picker
    return (fit_picker,)


@app.cell
def ols_parameter_recovery(fit_picker, fit_results, mo, np, ols_case):
    inspected_fit = fit_results[fit_picker.value]
    parameter_rows = [
        {
            "Term": name,
            "True coefficient": float(ols_case["true_coefficients"][index]),
            "OLS coefficient": float(inspected_fit["coefficients"][index]),
            "True mean contribution": float(ols_case["truth"][:, index].mean()),
            "Fitted mean contribution": float(inspected_fit["components"][:, index].mean()),
            "Contribution RMSE": float(np.sqrt(np.mean((inspected_fit["components"][:, index] - ols_case["truth"][:, index]) ** 2))),
            "True alpha": float(ols_case["alpha"][index - 1]) if 1 <= index <= ols_case["media"].shape[1] else None,
            "Fitted alpha": float(inspected_fit["alpha"][index - 1]) if 1 <= index <= ols_case["media"].shape[1] else None,
            "True half-saturation": float(ols_case["half"][index - 1]) if 1 <= index <= ols_case["media"].shape[1] else None,
            "Fitted half-saturation": float(inspected_fit["half"][index - 1]) if 1 <= index <= ols_case["media"].shape[1] else None,
            "Note": "Null effect: transform not identifiable" if 1 <= index <= ols_case["media"].shape[1] and ols_case["true_coefficients"][index] == 0 else "",
        }
        for index, name in enumerate(ols_case["names"])
    ]
    mo.vstack([mo.ui.table(parameter_rows, selection=None),
               mo.md("**Optimizer starts:**"), mo.ui.table(inspected_fit["attempts"], selection=None)
               if inspected_fit["attempts"] else mo.md("Fixed transforms: one linear least-squares solve.")])
    return (inspected_fit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prediction is not attribution

    For media (and separately for controls), we compute

    $$E = \frac{\sqrt{\sum_{t,j}(\widehat c_{tj}-c_{tj})^2}}{\sqrt{\sum_{t,j}c_{tj}^2}}.$$

    This uses the **individual contribution trajectories**, not their sum, their
    mean, the regression coefficients, or total sales. All active inputs count,
    including true-null channels; the intercept is excluded from both groups.
    Weeks 7–103 are scored. Weak true contributions make relative error sensitive.
    No denominator floor is added; an all-zero truth group has undefined error
    and cannot pass. Passing also requires solver convergence.

    Contributions can compensate for one another (and for the intercept), so a
    close prediction can conceal poor attribution. The component plots below show
    true and fitted time series on the **same axis**, without normalization.
    A 100% failure rate means all worlds missed the criterion, not 100% error in
    predicted sales. Means alone can also conceal time-varying errors.
    """)
    return


@app.cell
def contribution_plot_function(go, make_subplots, np):
    def plot_contribution_recovery(truth, components, target, prediction, first_week, names, term, title, *, media_spend=None):
        weeks = np.arange(first_week, first_week + len(truth))
        actual = truth[:, term]
        fitted = components[:, term]
        rmse = float(np.sqrt(np.mean((fitted - actual) ** 2)))
        denominator = np.linalg.norm(actual)
        relative = f"{np.linalg.norm(fitted - actual) / denominator:.1%}" if denominator > 0 else "undefined (true zero)"
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.25,
            specs=[[{}],[{"secondary_y":media_spend is not None}]],
            subplot_titles=("Outcome: fit can look good even with incorrect contributions",
                            f"{names[term]} contribution · RMSE {rmse:.4g} · relative L2 {relative}"))
        for values, name, color, dash in (
            (target, "Fit target", "gray", "solid"),
            (truth.sum(axis=1), "True noiseless outcome", "black", "dot"),
            (prediction, "Predicted outcome", "royalblue", "solid"),
        ):
            fig.add_trace(go.Scatter(x=weeks, y=values, name=name, mode="lines",
                line={"color":color,"dash":dash}, legend="legend"),row=1,col=1)
        fig.add_trace(go.Scatter(x=weeks,y=actual,name=f"True {names[term]}",mode="lines",
            line={"color":"black","width":2},legend="legend2"),row=2,col=1)
        fig.add_trace(go.Scatter(x=weeks,y=fitted,name=f"Estimated {names[term]}",mode="lines",
            line={"color":"darkorange","width":2,"dash":"dash"},legend="legend2"),row=2,col=1)
        fig.update_yaxes(title_text="Outcome units",zeroline=True)
        if media_spend is not None:
            spend=np.asarray(media_spend,dtype=float)
            if spend.shape!=weeks.shape or not np.isfinite(spend).all() or np.any(spend<0):
                raise ValueError("Media spend must be a finite nonnegative vector aligned to the plotted weeks")
            fig.add_trace(go.Scatter(x=weeks,y=spend,name=f"Raw spend {names[term]} · right axis",mode="lines",
                line={"color":"#8e44ad","width":1.4,"dash":"dot"},opacity=0.65,legend="legend2"),row=2,col=1,secondary_y=True)
            fig.update_yaxes(title_text="Raw media spend (original input units)",rangemode="tozero",
                showgrid=False,zeroline=False,color="#8e44ad",row=2,col=1,secondary_y=True)
        fig.update_xaxes(title_text="Reported week",row=2,col=1)
        fig.update_layout(title={"text":title,"font":{"size":16}},template="plotly_white",
            height=820,margin={"t":115,"b":120},
            legend={"orientation":"h","y":0.54,"yanchor":"top"},
            legend2={"orientation":"h","y":-0.15,"yanchor":"top"})
        return fig

    return (plot_contribution_recovery,)


@app.cell
def ols_recovery_plot(
    fit_world_label,
    go,
    inspected_fit,
    make_subplots,
    mo,
    np,
    ols_case,
):
    fit_figure = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.24,
                               subplot_titles=("Outcome fit (same fitted window)", "Mean contribution by term"))
    fit_weeks = np.arange(ols_case["start"], len(ols_case["observed"]))
    fit_figure.add_trace(go.Scatter(x=fit_weeks, y=inspected_fit["target"], name="Fit target",
                                   mode="lines", line={"color": "gray"}, legend="legend"), row=1,col=1)
    fit_figure.add_trace(go.Scatter(x=fit_weeks, y=ols_case["noiseless"][ols_case["start"]:], name="True noiseless outcome",
                                   mode="lines", line={"color": "black", "dash": "dot"}, legend="legend"), row=1,col=1)
    fit_figure.add_trace(go.Scatter(x=fit_weeks, y=inspected_fit["prediction"], name="OLS prediction",
                                   mode="lines", line={"color": "royalblue"}, legend="legend"), row=1,col=1)
    fit_figure.add_trace(go.Bar(x=ols_case["names"], y=ols_case["truth"].mean(axis=0), name="True mean contribution", legend="legend2"),row=2,col=1)
    fit_figure.add_trace(go.Bar(x=ols_case["names"], y=inspected_fit["components"].mean(axis=0), name="Fitted mean contribution", legend="legend2"),row=2,col=1)
    fit_figure.update_xaxes(matches=None, row=2,col=1)
    fit_figure.update_xaxes(title_text="Reported week", row=1,col=1)
    fit_figure.update_layout(title=f"{fit_world_label}<br><sup>{inspected_fit['label']}</sup>",
        height=850, template="plotly_white", barmode="group", margin={"t":100,"b":120},
        legend={"orientation":"h","y":0.54,"yanchor":"top"},
        legend2={"orientation":"h","y":-0.16,"yanchor":"top"})
    mo.ui.plotly(fit_figure)
    return


@app.cell
def current_contribution_picker(mo, np, ols_case):
    current_component_picker = mo.ui.dropdown(
        options={name:index for index,name in enumerate(ols_case["names"]) if name=="Intercept" or np.any(ols_case["truth"][:,index]!=0)},
        value=next((name for index,name in enumerate(ols_case["names"]) if name!="Intercept" and np.any(ols_case["truth"][:,index]!=0)), "Intercept"), label="Inspect contribution (C = media, Z = control)",
    )
    current_component_picker
    return (current_component_picker,)


@app.cell
def current_contribution_timeseries(
    current_component_picker,
    fit_world_label,
    inspected_fit,
    mo,
    ols_case,
    plot_contribution_recovery,
):
    current_contribution_figure = plot_contribution_recovery(
        ols_case["truth"], inspected_fit["components"], inspected_fit["target"],
        inspected_fit["prediction"], ols_case["start"], ols_case["names"],
        current_component_picker.value, f"{fit_world_label}<br><sup>{inspected_fit['label']}</sup>",
    )
    mo.ui.plotly(current_contribution_figure)
    return


@app.cell
def ols_numerical_checks(
    fit_profiled_ols,
    mo,
    np,
    showcase_cfg,
    solve_linear_ols,
):
    def check_ols_numerics():
        # Small deterministic test fixture, not another generated SCM world.
        rng = np.random.default_rng(17)
        media = rng.gamma(2.0, 2.0, size=(80, 2))
        controls = rng.normal(size=(80, 1))
        alpha = np.array([0.3, 0.65])
        half = np.array([2.5, 5.0])
        transformed = []
        for channel in range(2):
            kernel = alpha[channel] ** np.arange(showcase_cfg.l_max)
            kernel /= kernel.sum()
            carried = np.convolve(media[:, channel], kernel)[:len(media)]
            transformed.append(carried / (half[channel] + carried))
        design = np.column_stack([np.ones(len(media)), *transformed, controls])
        coefficient = np.array([3.0, 1.5, -0.8, 0.4])
        target = design @ coefficient
        original = media.copy()
        for fixed in ((alpha, half), None):
            result = fit_profiled_ols(media, controls, target, 7, fixed=fixed)
            assert result["success"], result["message"]
            assert result["rank"] == 4
            np.testing.assert_allclose(result["prediction"], target[7:], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(result["coefficients"], coefficient, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(result["alpha"], alpha, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(result["half"], half, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(result["components"].sum(axis=1), result["prediction"], rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(media, original)
        # A deliberately aliased design must expose rank deficiency, not claim uniqueness.
        duplicate = np.column_stack([design, design[:, 1]])
        _, prediction, rank, condition = solve_linear_ols(duplicate, target)
        assert rank < duplicate.shape[1] and np.isinf(condition)
        np.testing.assert_allclose(prediction, target, rtol=1e-12, atol=1e-12)
        return "Synthetic fixed/estimated-transform recovery, input preservation and rank-deficiency checks passed."

    ols_numerical_check = check_ols_numerics()
    mo.md(ols_numerical_check)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Bounded 50-world study

    **50 new cells × one world**, streamed as 25 two-cell batches, seeds 10029–10053.
    This studies the current **constant-baseline showcase recipe**, with four weighted
    active-count combinations—not every configuration the generator supports.
    Each world receives all four fits, using the existing two-start/120-evaluation
    budget and the common 97-week window. Generation remains the naive library call.

    At each threshold (5%, 10%), media success requires **solver convergence AND
    media relative L2 ≤ threshold**; controls use the same rule. Joint success
    requires convergence AND both errors ≤ threshold. Nonconvergence or undefined
    errors never count as successes. Fixed-transform OLS has no iterative optimizer;
    a completed finite linear solve meets the convergence condition. Rank deficiency
    is reported separately. These are groupwise errors over the component matrices,
    not a requirement that every individual channel meet the threshold.

    Each corpus batch and every world's fit results are checkpointed under
    `notebooks/ols_study_50_seed10029/`. A compatible rerun resumes saved work;
    completed studies load without generating or fitting again. Changing the recipe
    or fitting implementation requires a new study directory. Unexpected errors
    stop execution instead of silently dropping worlds from the denominator.
    """)
    return


@app.cell
def study_imports():
    import json as study_json
    import hashlib as study_hashlib
    import inspect as study_inspect
    from pathlib import Path as StudyPath
    from dataclasses import asdict as study_asdict, replace

    return (
        StudyPath,
        replace,
        study_asdict,
        study_hashlib,
        study_inspect,
        study_json,
    )


@app.cell
def study_functions(
    StudyPath,
    compile_media_transform,
    fit_oracle_graph_ols,
    fit_profiled_ols,
    mo,
    np,
    perf_counter,
    pg,
    prepare_ols_case,
    pytensor,
    relative_component_error,
    replace,
    run_four_ols_fits,
    solve_linear_ols,
    study_asdict,
    study_hashlib,
    study_inspect,
    study_json,
    transform_media_columns,
):
    def study_json_safe(value):
        if isinstance(value, dict):
            return {str(key): study_json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [study_json_safe(item) for item in value]
        if isinstance(value, (float, np.floating)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.ndarray):
            return study_json_safe(value.tolist())
        return value


    def write_study_json(path, data):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(study_json.dumps(study_json_safe(data), indent=2, allow_nan=False))
        temporary.replace(path)


    def summarize_study(records):
        rows = []
        labels = list(dict.fromkeys(fit["label"] for record in records for fit in record["fits"]))
        for label in labels:
            fits = [fit for record in records for fit in record["fits"] if fit["label"] == label]
            for threshold in (0.05, 0.10):
                media_ok = [bool(fit["success"] and fit["media_error"] is not None and np.isfinite(fit["media_error"]) and fit["media_error"] <= threshold) for fit in fits]
                control_ok = [bool(fit["success"] and fit["control_error"] is not None and np.isfinite(fit["control_error"]) and fit["control_error"] <= threshold) for fit in fits]
                joint = sum(m and c for m, c in zip(media_ok, control_ok, strict=True))
                count = len(fits)
                rows.append({"Fit": label, "Threshold": f"{threshold:.0%}",
                    "Worlds completed / planned": f"{count} / 50",
                    "Converged": sum(fit["success"] for fit in fits),
                    "Rank deficient": sum(fit["rank"] < fit["columns"] for fit in fits),
                    "Media successes": sum(media_ok), "Media failure %": 100 * (count - sum(media_ok)) / count,
                    "Control successes": sum(control_ok), "Control failure %": 100 * (count - sum(control_ok)) / count,
                    "Joint successes": joint, "Joint failures": count - joint,
                    "Joint failure %": 100 * (count - joint) / count})
        return rows


    def run_ols_study(base_cfg, directory):
        directory = StudyPath(directory)
        source = "\n".join(study_inspect.getsource(function) for function in (
            compile_media_transform, transform_media_columns, solve_linear_ols,
            fit_profiled_ols, prepare_ols_case, run_four_ols_fits, relative_component_error, fit_oracle_graph_ols,
        ))
        manifest = study_json_safe({"oracle_graph": True, "worlds": 50, "cells_per_batch": 2, "draws_per_cell": 1,
            "first_seed": 10029, "thresholds": [0.05, 0.1], "config": study_asdict(base_cfg),
            "fit_source_sha256": study_hashlib.sha256(source.encode()).hexdigest(),
            "pytensor_version": pytensor.__version__, "numpy_version": np.__version__,
            "pymc_generator_version": pg.__version__})
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            if study_json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("Study recipe/implementation differs from checkpoint; use a new directory")
        else:
            if any(directory.iterdir()):
                raise ValueError("Refusing an existing nonempty directory without a study manifest")
            write_study_json(manifest_path, manifest)
            (directory / "fit_implementation.py.txt").write_text(source)
        records = []
        generation_seconds = 0.0
        began = perf_counter()
        with mo.status.progress_bar(total=50, title="Generate and fit 50 worlds") as progress:
            for batch_id in range(25):
                cfg = replace(base_cfg, n_cells=2, draws_per_cell=1, seed=10029 + batch_id)
                corpus_path = directory / f"batch_{batch_id:02d}.npz"
                generation_path = directory / f"batch_{batch_id:02d}_timing.json"
                if corpus_path.exists():
                    corpus = pg.load_corpus(corpus_path)
                    if not generation_path.exists():
                        raise ValueError(f"Missing timing checkpoint for {corpus_path}")
                    generation_time = study_json.loads(generation_path.read_text())["generation_seconds"]
                else:
                    started = perf_counter()
                    corpus = pg.sample_prior_predictive(cfg)
                    generation_time = perf_counter() - started
                    errors = pg.DataGenerator.validate_corpus(corpus)
                    if errors:
                        raise ValueError(f"Invalid generated corpus: {errors}")
                    assert len(corpus["cell_id"]) == 2 and len(np.unique(corpus["cell_id"])) == 2
                    assert np.all(np.ptp(corpus["baseline_intrinsic"], axis=1) == 0)
                    write_study_json(generation_path, {"seed": cfg.seed, "generation_seconds": generation_time})
                    pg.save_corpus(corpus, corpus_path)
                generation_seconds += generation_time
                for world in range(2):
                    world_id = batch_id * 2 + world
                    record_path = directory / f"world_{world_id:02d}.json"
                    arrays_path = directory / f"world_{world_id:02d}_fits.npz"
                    if record_path.exists() and arrays_path.exists():
                        record = study_json.loads(record_path.read_text())
                        assert record["world_id"] == world_id and record["seed"] == cfg.seed
                        assert len(record["fits"]) == 4 and arrays_path.exists()
                    else:
                        case = prepare_ols_case(corpus, world, cfg)
                        results = run_four_ols_fits(case)
                        n_media = case["media"].shape[1]
                        record = {"world_id": world_id, "batch": batch_id, "seed": cfg.seed,
                            "cell_id": int(corpus["cell_id"][world]), "local_world": world,
                            "media": n_media, "controls": case["controls"].shape[1],
                            "nonzero_media_effects": int(np.count_nonzero(case["true_coefficients"][1:1+n_media])),
                            "nonzero_control_effects": int(np.count_nonzero(case["true_coefficients"][1+n_media:])),
                            "fits_run": True, "oracle_graph": True,
                            "fitted_weeks": len(case["truth"]), "first_week": case["start"],
                            "terms": case["names"], "fits": []}
                        arrays = {"truth": case["truth"], "true_coefficients": case["true_coefficients"],
                                  "true_alpha": case["alpha"], "true_half": case["half"]}
                        for index, fit in enumerate(results):
                            record["fits"].append({key: fit[key] for key in (
                                "label", "success", "message", "rank", "columns", "condition",
                                "seconds", "residual_calls", "solver_nfev", "boundary_count", "attempts")})
                            record["fits"][-1].update(
                                media_error=relative_component_error(fit["components"][:, 1:1+n_media], case["truth"][:, 1:1+n_media]),
                                control_error=relative_component_error(fit["components"][:, 1+n_media:], case["truth"][:, 1+n_media:]),
                                target_rmse=float(np.sqrt(np.mean((fit["prediction"] - fit["target"]) ** 2))),
                                noiseless_rmse=float(np.sqrt(np.mean((fit["prediction"] - case["noiseless"][case["start"]:]) ** 2))),
                            )
                            for key in ("coefficients", "alpha", "half", "prediction", "target", "components"):
                                arrays[f"fit_{index}_{key}"] = fit[key]
                        np.savez_compressed(arrays_path, **arrays)
                        record = study_json_safe(record)
                        write_study_json(record_path, record)
                    records.append(record)
                    fitting_seconds = sum(fit["seconds"] for item in records for fit in item["fits"])
                    report = {"completed_worlds": len(records), "planned_worlds": 50,
                        "complete": len(records) == 50, "generation_seconds": generation_seconds,
                        "fitting_seconds": fitting_seconds, "elapsed_this_invocation": perf_counter()-began,
                        "summary": summarize_study(records)}
                    write_study_json(directory / "progress.json", report)
                    progress.update(increment=1, subtitle=f"{len(records)}/50 worlds; generation {generation_seconds:.1f}s; fitting {fitting_seconds:.1f}s")
                    mo.output.replace(mo.vstack([
                        mo.md(f"### Study progress: {len(records)}/50 worlds"),
                        mo.ui.table(report["summary"], selection=None),
                    ]))
                    print(f"Completed world {world_id+1}/50 (seed {cfg.seed}); generation {generation_seconds:.2f}s; fitting {fitting_seconds:.2f}s", flush=True)
        assert len(records) == 50
        assert len({(r["seed"], r["cell_id"]) for r in records}) == 50
        write_study_json(directory / "records.json", records)
        return records, report

    return run_ols_study, study_json_safe, write_study_json


@app.cell
def study_run_control(mo):
    study_run_button = mo.ui.run_button(label="Run / resume the bounded 50-world study")
    study_run_button
    return (study_run_button,)


@app.cell
def run_fifty_world_study(
    constant_baseline_cfg,
    mo,
    run_ols_study,
    study_run_button,
):
    mo.stop(not study_run_button.value, mo.md("The 50-world study runs only on request. Checkpoints survive a kernel restart."))
    study_records, study_report = run_ols_study(constant_baseline_cfg, "notebooks/ols_study_50_seed10029")
    return (study_report,)


@app.cell
def fifty_world_summary(mo, study_report):
    mo.vstack([
        mo.md(f"### 50-world study — {study_report['completed_worlds']}/50 completed\n\n"
              f"Generation: **{study_report['generation_seconds']:.1f} s** · "
              f"four-variant fitting: **{study_report['fitting_seconds']:.1f} s**. "
              "Generation time includes compilation, candidate draws and filtering; it is not compilation-only timing."),
        mo.ui.table(study_report["summary"], selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Inspect contribution recovery in the 50-world study

    Choose a saved study world, a fitting variant, and an individual contribution.
    These world IDs belong to the study, not the earlier exploratory batches.
    Selections only load saved results; they never generate or refit worlds.
    The default fit is **true transformations + observation noise**.
    """)
    return


@app.cell
def saved_study_selectors(StudyPath, mo, study_json):
    saved_study_records = [
        study_json.loads(path.read_text())
        for path in sorted(StudyPath("notebooks/ols_study_50_seed10029").glob("world_[0-9][0-9].json"))
    ]
    mo.stop(not saved_study_records or any(len(r.get("fits", [])) != 4 for r in saved_study_records), mo.md("Current datasets regenerated; OLS fits pending."))
    study_world_options = {
        f"Study world {record['world_id'] + 1} · seed {record['seed']} · {record['media']} media / {record['controls']} controls": record["world_id"]
        for record in saved_study_records
    }
    study_world_picker = mo.ui.dropdown(options=study_world_options,
        value=next(iter(study_world_options)),label="Saved world")
    study_fit_options = {fit["label"]: index for index,fit in enumerate(saved_study_records[0]["fits"])}
    study_fit_picker = mo.ui.dropdown(options=study_fit_options,
        value=list(study_fit_options)[2],label="Saved fit")
    mo.vstack([study_world_picker,study_fit_picker])
    return saved_study_records, study_fit_picker, study_world_picker


@app.cell
def current_generation_summary(mo, saved_study_records):
    current_generation_rows=[{"World":r["world_id"]+1,"Media inputs":r["media"],"Contributing media":r["nonzero_media_effects"],
        "Control inputs":r["controls"],"Contributing controls":r["nonzero_control_effects"]} for r in saved_study_records]
    mo.vstack([mo.md("### Current regenerated study: 50 worlds\n\nThese are the datasets used by the rerun analyses below. Effect counts are no longer fixed at five/three."),mo.ui.table(current_generation_rows,selection=None)])
    return


@app.cell
def saved_fit_restriction(mo):
    fit_restriction_picker = mo.ui.dropdown(
        options=["Original", "Saturation constrained"], value="Saturation constrained",
        label="Compare fitting restriction",
    )
    fit_restriction_picker
    return (fit_restriction_picker,)


@app.cell
def saved_study_contributions(
    StudyPath,
    fit_restriction_picker,
    mo,
    np,
    study_fit_picker,
    study_json,
    study_world_picker,
):
    viewed_study_directory = StudyPath("notebooks/ols_study_50_seed10029_saturation95" if fit_restriction_picker.value == "Saturation constrained" else "notebooks/ols_study_50_seed10029")
    viewed_study_record = study_json.loads((viewed_study_directory / f"world_{study_world_picker.value:02d}.json").read_text())
    viewed_study_fit = viewed_study_record["fits"][study_fit_picker.value]
    with np.load(viewed_study_directory / f"world_{study_world_picker.value:02d}_fits.npz",allow_pickle=False) as _archive:
        viewed_study_arrays = {key: _archive[key] for key in _archive.files}
    study_component_picker = mo.ui.dropdown(
        options={name:index for index,name in enumerate(viewed_study_record["terms"]) if name=="Intercept" or np.any(viewed_study_arrays["truth"][:,index]!=0)},
        value=next((name for index,name in enumerate(viewed_study_record["terms"]) if name!="Intercept" and np.any(viewed_study_arrays["truth"][:,index]!=0)), "Intercept"),label="Contribution (C = media, Z = control)",
    )
    mo.vstack([
        mo.md(f"**Converged:** {viewed_study_fit['success']} · "
              f"**Media group error:** {viewed_study_fit['media_error']:.1%} · "
              f"**Control group error:** {viewed_study_fit['control_error']:.1%}\n\n"
              f"{viewed_study_fit['message']}"),
        study_component_picker,
    ])
    return (
        study_component_picker,
        viewed_study_arrays,
        viewed_study_directory,
        viewed_study_fit,
        viewed_study_record,
    )


@app.cell
def saved_study_contribution_plot(
    fit_restriction_picker,
    mo,
    plot_contribution_recovery,
    study_component_picker,
    study_fit_picker,
    viewed_study_arrays,
    viewed_study_fit,
    viewed_study_record,
):
    study_contribution_figure = plot_contribution_recovery(
        viewed_study_arrays["truth"], viewed_study_arrays[f"fit_{study_fit_picker.value}_components"],
        viewed_study_arrays[f"fit_{study_fit_picker.value}_target"],
        viewed_study_arrays[f"fit_{study_fit_picker.value}_prediction"],
        viewed_study_record["first_week"], viewed_study_record["terms"],study_component_picker.value,
        f"Study world {viewed_study_record['world_id']+1} · seed {viewed_study_record['seed']}<br><sup>{viewed_study_fit['label']} · {fit_restriction_picker.value}</sup>",
    )
    mo.ui.plotly(study_contribution_figure)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Practical contribution recovery: relative OR outcome-scaled tolerance

    For each group (media, controls), passing requires convergence and

    $$\|\widehat C-C\|_F \leq \max(r\|C\|_F,\ a\|y_{\mathrm{noiseless}}\|_2).$$

    We cross **r = 5%, 10%** with **a = 0.5%, 1%**. Joint success requires both
    groups to pass. The intercept is excluded from the groups but remains part of
    the noiseless outcome scale. This is an absolute-error allowance in outcome
    units, not dropping low-contribution channels. Large spurious effects on
    true-zero channels still count as errors. Group errors retain individual
    channel columns, so opposing channel errors cannot cancel.

    The same 97 fitted weeks and saved fits are used. The original percentage-only
    report above is preserved; **no new generation or fitting occurs**.
    """)
    return


@app.cell
def practical_scoring_functions(StudyPath, np):
    def practical_group_pass(error_norm, truth_norm, outcome_norm, converged, relative, absolute):
        values = (error_norm, truth_norm, outcome_norm)
        return bool(converged and all(np.isfinite(value) and value >= 0 for value in values)
                    and error_norm <= max(relative * truth_norm, absolute * outcome_norm))


    def rescore_practical_recovery(records, directory):
        scored = []
        for record in records:
            path = StudyPath(directory) / f"world_{record['world_id']:02d}_fits.npz"
            with np.load(path, allow_pickle=False) as arrays:
                truth = arrays["truth"]
                # Variant 4's target is exactly observed outcome minus stored noise.
                outcome_norm = float(np.linalg.norm(arrays["fit_3_target"]))
                n_media = record["media"]
                for index, fit in enumerate(record["fits"]):
                    estimate = arrays[f"fit_{index}_components"]
                    media_truth = float(np.linalg.norm(truth[:, 1:1+n_media]))
                    control_truth = float(np.linalg.norm(truth[:, 1+n_media:]))
                    media_error = float(np.linalg.norm(estimate[:, 1:1+n_media] - truth[:, 1:1+n_media]))
                    control_error = float(np.linalg.norm(estimate[:, 1+n_media:] - truth[:, 1+n_media:]))
                    for relative in (0.05, 0.10):
                        for absolute in (0.005, 0.01):
                            media_pass = practical_group_pass(media_error, media_truth, outcome_norm, fit["success"], relative, absolute)
                            control_pass = practical_group_pass(control_error, control_truth, outcome_norm, fit["success"], relative, absolute)
                            scored.append({"world_id":record["world_id"], "fit":fit["label"],
                                "relative":relative, "absolute":absolute, "converged":fit["success"],
                                "media_error_norm":media_error, "control_error_norm":control_error,
                                "media_truth_norm":media_truth, "control_truth_norm":control_truth,
                                "outcome_norm":outcome_norm,
                                "media_pass":media_pass, "control_pass":control_pass,
                                "joint_pass":media_pass and control_pass})
        summary = []
        for label in dict.fromkeys(row["fit"] for row in scored):
            for relative in (0.05, 0.10):
                for absolute in (0.005, 0.01):
                    selected = [row for row in scored if row["fit"] == label and row["relative"] == relative and row["absolute"] == absolute]
                    total = len(selected)
                    summary.append({"Fit":label, "Relative tolerance":f"{relative:.0%}",
                        "Outcome-scaled tolerance":f"{absolute:.1%}", "Worlds":total,
                        "Converged":sum(row["converged"] for row in selected),
                        "Media successes":sum(row["media_pass"] for row in selected),
                        "Control successes":sum(row["control_pass"] for row in selected),
                        "Joint successes":sum(row["joint_pass"] for row in selected),
                        "Media failure %":100 * sum(not row["media_pass"] for row in selected)/total,
                        "Control failure %":100 * sum(not row["control_pass"] for row in selected)/total,
                        "Joint failure %":100 * sum(not row["joint_pass"] for row in selected)/total})
        return scored, summary


    # Boundary and counterexample checks: tiny errors pass; large null effects do not.
    assert practical_group_pass(0.5, 0.0, 100.0, True, 0.05, 0.005)
    assert not practical_group_pass(0.5001, 0.0, 100.0, True, 0.05, 0.005)
    assert practical_group_pass(0.5001, 0.0, 100.0, True, 0.05, 0.01)
    assert practical_group_pass(5.0, 100.0, 1.0, True, 0.05, 0.005)
    assert not practical_group_pass(0.0, 1.0, 100.0, False, 0.1, 0.01)
    assert not practical_group_pass(268.0, 0.0, 100.0, True, 0.1, 0.01)
    assert not practical_group_pass(float("nan"), 1.0, 100.0, True, 0.1, 0.01)
    return practical_group_pass, rescore_practical_recovery


@app.cell
def practical_recovery_summary(
    StudyPath,
    mo,
    rescore_practical_recovery,
    saved_study_records,
    write_study_json,
):
    practical_scores, practical_summary = rescore_practical_recovery(
        saved_study_records, "notebooks/ols_study_50_seed10029",
    )
    assert len(practical_scores) == len(saved_study_records) * 4 * 4
    write_study_json(StudyPath("notebooks/ols_study_50_seed10029/practical_scoring.json"), {
        "definition":"converged and error_norm <= max(relative * truth_norm, absolute * noiseless_outcome_norm)",
        "relative_thresholds":[0.05,0.1], "outcome_scaled_thresholds":[0.005,0.01],
        "completed_worlds":len(saved_study_records), "summary":practical_summary, "scores":practical_scores,
    })
    mo.ui.table(practical_summary, selection=None)
    return (practical_summary,)


@app.cell
def selected_practical_recovery(
    mo,
    rescore_practical_recovery,
    viewed_study_directory,
    viewed_study_fit,
    viewed_study_record,
):
    viewed_practical_scores = [
        {
            "Relative tolerance":f"{row['relative']:.0%}",
            "Outcome-scaled tolerance":f"{row['absolute']:.1%}",
            "Media error / outcome":f"{row['media_error_norm']/row['outcome_norm']:.2%}",
            "Control error / outcome":f"{row['control_error_norm']/row['outcome_norm']:.2%}",
            "Media pass":row["media_pass"], "Control pass":row["control_pass"], "Joint pass":row["joint_pass"],
        }
        for row in rescore_practical_recovery([viewed_study_record], viewed_study_directory)[0]
        if row["world_id"] == viewed_study_record["world_id"] and row["fit"] == viewed_study_fit["label"]
    ]
    mo.vstack([mo.md(f"### Practical scoring for the saved-world viewer\n\nStudy world {viewed_study_record['world_id']+1} · {viewed_study_fit['label']}"),
               mo.ui.table(viewed_practical_scores, selection=None)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saturation-constrained comparison

    For the **two estimated-transformation variants only**, every active channel
    must have response ≤0.95 in at least 10% of the 97 fitted weeks (at least 10
    weeks). Adstock still uses the full input history. We enforce the bound using
    the tenth-smallest fitted-week adstock value and a decay-dependent lower bound
    on half-saturation. This is the discrete version of the 10th-percentile rule.

    We preserve the original numerical half-saturation range, raising its lower
    bound only when the new restriction requires it. Optimization interpolates
    between those bounds in log space. OLS coefficients remain unconstrained;
    solver budgets and starting proposals are unchanged. This is a different
    parameterization, so optimizer paths can differ. No true effects are used in
    optimization. True-parameter feasibility is checked separately for diagnosis.

    Original fits and the fixed-transform variants are preserved. This comparison
    uses the same 50 saved worlds and saves separate constrained checkpoints.
    """)
    return


@app.cell
def constraint_adstock_compilation(mechanisms, pt, pytensor, showcase_cfg):
    def compile_constraint_adstock(l_max):
        media = pt.dvector("constraint_media")
        alpha = pt.dscalar("constraint_alpha")
        carried = mechanisms.apply_geometric_carryover(media[:, None], alpha, l_max)[:, 0]
        return pytensor.function([media, alpha], carried, name="constraint_adstock", mode="FAST_RUN")

    constraint_adstock = compile_constraint_adstock(showcase_cfg.l_max)
    return compile_constraint_adstock, constraint_adstock


@app.cell
def saturation_constrained_solver(
    constraint_adstock,
    least_squares,
    np,
    perf_counter,
    solve_linear_ols,
    transform_media_columns,
):
    def saturation_half_lower_bound(media, alpha, start, input_scale):
        carried = np.column_stack([constraint_adstock(media[:, k], alpha[k]) for k in range(media.shape[1])])[start:]
        required = int(np.ceil(0.1 * len(carried)))
        quantile = np.partition(carried, required - 1, axis=0)[required - 1]
        # Tiny arithmetic safety margin, not a relaxed saturation constraint.
        return np.maximum(input_scale * 0.001, quantile * (0.05 / 0.95) * (1 + 1e-12))


    def fit_saturation_constrained_ols(media, controls, target, start, *, starts=2, max_nfev=120):
        began = perf_counter()
        n_media = media.shape[1]
        input_scale = np.array([np.mean(column[column > 0]) if np.any(column > 0) else 1.0 for column in media.T])
        half_upper = input_scale * 1000.0
        target_scale = max(float(np.std(target[start:])), 1e-8)
        calls = 0

        def decode(theta):
            alpha = theta[:n_media]
            half_lower = saturation_half_lower_bound(media, alpha, start, input_scale)
            if np.any(half_lower >= half_upper):
                raise ValueError("Saturation constraint conflicts with numerical half-saturation upper bound")
            half = np.exp(np.log(half_lower) + theta[n_media:] * (np.log(half_upper) - np.log(half_lower)))
            return alpha, half

        def solve(theta):
            alpha, half = decode(theta)
            transformed = transform_media_columns(media, alpha, half)[start:]
            design = np.column_stack([np.ones(len(transformed)), transformed, controls[start:]])
            coef, prediction, rank, condition = solve_linear_ols(design, target[start:])
            return alpha, half, transformed, design, coef, prediction, rank, condition

        def residual(theta):
            nonlocal calls
            calls += 1
            return (solve(theta)[5] - target[start:]) / target_scale

        rng = np.random.default_rng(2026)
        candidates, attempts = [], []
        for attempt in range(starts):
            alpha0 = np.full(n_media, 0.5) if attempt == 0 else rng.uniform(0.15, 0.85, n_media)
            half0 = input_scale if attempt == 0 else input_scale * np.exp(rng.uniform(-1, 1, n_media))
            half_lower = saturation_half_lower_bound(media, alpha0, start, input_scale)
            fraction = np.clip((np.log(half0) - np.log(half_lower)) / (np.log(half_upper) - np.log(half_lower)), 1e-8, 1-1e-8)
            result = least_squares(residual, np.r_[alpha0, fraction], bounds=(np.zeros(2*n_media), np.ones(2*n_media)),
                                   max_nfev=max_nfev, ftol=1e-8, xtol=1e-8, gtol=1e-8)
            candidates.append(result)
            attempts.append({"start":attempt+1,"success":bool(result.success),"nfev":int(result.nfev),
                             "cost":float(result.cost),"message":str(result.message)})
        best = min(candidates, key=lambda result: result.cost)
        alpha, half, transformed, design, coef, prediction, rank, condition = solve(best.x)
        fraction_below = np.mean(transformed <= 0.95 + 1e-12, axis=0)
        assert np.all(fraction_below >= 0.1), "Returned solution violates the saturation constraint"
        return {"alpha":alpha,"half":half,"coefficients":coef,"prediction":prediction,
            "components":design*coef,"rank":rank,"columns":design.shape[1],"condition":condition,
            "success":bool(best.success),"message":str(best.message),"solver_nfev":sum(a["nfev"] for a in attempts),
            "residual_calls":calls,"boundary_count":int(np.count_nonzero(best.active_mask)),
            "attempts":attempts,"seconds":perf_counter()-began,"constraint_fraction":fraction_below}


    # Validate the bound independently of fitting, including zero inputs and alpha endpoints.
    for _alpha in (0.0, 0.5, 1.0):
        _media = np.column_stack([np.linspace(0, 5, 104), np.ones(104), np.zeros(104)])
        _decay = np.full(3, _alpha)
        _lower = saturation_half_lower_bound(_media, _decay, 7, np.ones(3))
        _response = transform_media_columns(_media, _decay, _lower)[7:]
        assert np.all(np.mean(_response <= 0.95 + 1e-12, axis=0) >= 0.1)
        _safe_response = transform_media_columns(_media, _decay, _lower * 100)[7:]
        assert np.all(np.mean(_safe_response <= 0.95 + 1e-12, axis=0) >= 0.1)
    return fit_saturation_constrained_ols, saturation_half_lower_bound


@app.cell
def constrained_comparison_runner(
    StudyPath,
    compile_constraint_adstock,
    constant_baseline_cfg,
    fit_oracle_graph_ols,
    fit_saturation_constrained_ols,
    mo,
    np,
    pg,
    prepare_ols_case,
    relative_component_error,
    replace,
    saturation_half_lower_bound,
    solve_linear_ols,
    study_asdict,
    study_hashlib,
    study_inspect,
    study_json,
    study_json_safe,
    transform_media_columns,
    write_study_json,
):
    def run_saturation_comparison(records, source_dir, destination):
        source_dir, destination = StudyPath(source_dir), StudyPath(destination)
        source_manifest = study_json.loads((source_dir / "manifest.json").read_text())
        if study_json_safe(study_asdict(constant_baseline_cfg)) != source_manifest["config"]:
            raise ValueError("Current config differs from the saved study")
        implementation = "\n".join(study_inspect.getsource(f) for f in (
            compile_constraint_adstock, saturation_half_lower_bound, fit_saturation_constrained_ols,
            solve_linear_ols, transform_media_columns, prepare_ols_case, fit_oracle_graph_ols))
        source_hashes = {path.name:study_hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted(source_dir.glob("world_*"))}
        manifest = {"source_manifest":source_manifest,"source_hashes":source_hashes,
                    "constraint":{"saturation_max":0.95,"minimum_week_fraction":0.1},
                    "implementation_sha256":study_hashlib.sha256(implementation.encode()).hexdigest()}
        destination.mkdir(parents=True,exist_ok=True)
        if (destination / "manifest.json").exists():
            if study_json.loads((destination / "manifest.json").read_text()) != manifest:
                raise ValueError("Constrained comparison differs from saved checkpoint")
        else:
            if any(destination.iterdir()):
                raise ValueError("Nonempty comparison directory without manifest")
            write_study_json(destination / "manifest.json",manifest)
            (destination / "implementation.py.txt").write_text(implementation)
        output = []
        with mo.status.progress_bar(total=len(records),title="Constrained refits (no generation)") as progress:
            for original in records:
                world_id = original["world_id"]
                record_path = destination / f"world_{world_id:02d}.json"
                if record_path.exists():
                    record = study_json.loads(record_path.read_text())
                    assert (destination / f"world_{world_id:02d}_fits.npz").exists()
                else:
                    corpus = pg.load_corpus(source_dir / f"batch_{original['batch']:02d}.npz")
                    cfg = replace(constant_baseline_cfg, seed=original["seed"])
                    case = prepare_ols_case(corpus, original["local_world"], cfg)
                    with np.load(source_dir / f"world_{world_id:02d}_fits.npz",allow_pickle=False) as saved:
                        arrays = {key:saved[key] for key in saved.files}
                    record = study_json.loads(study_json.dumps(original))
                    true_response = transform_media_columns(case["media"],case["alpha"],case["half"])[case["start"]:]
                    true_fraction = np.mean(true_response <= 0.95 + 1e-12,axis=0)
                    record["true_constraint_fraction"] = true_fraction.tolist()
                    record["true_nonnull_constraint_violations"] = int(np.count_nonzero(
                        (true_fraction < 0.1) & (case["true_coefficients"][1:1+original["media"]] != 0)))
                    for index, target in enumerate((case["observed"],case["noiseless"])):
                        fit = fit_oracle_graph_ols(case,target,solver=fit_saturation_constrained_ols)
                        n = original["media"]
                        updated = {key:fit[key] for key in ("success","message","rank","columns","condition",
                            "seconds","residual_calls","solver_nfev","boundary_count","attempts")}
                        updated.update(label=original["fits"][index]["label"],
                            media_error=relative_component_error(fit["components"][:,1:1+n],case["truth"][:,1:1+n]),
                            control_error=relative_component_error(fit["components"][:,1+n:],case["truth"][:,1+n:]),
                            target_rmse=float(np.sqrt(np.mean((fit["prediction"]-target[case["start"]:])**2))),
                            noiseless_rmse=float(np.sqrt(np.mean((fit["prediction"]-case["noiseless"][case["start"]:])**2))),
                            constraint_fraction=fit["constraint_fraction"].tolist())
                        record["fits"][index] = updated
                        for key in ("coefficients","alpha","half","prediction","components","constraint_fraction"):
                            arrays[f"fit_{index}_{key}"] = fit[key]
                    np.savez_compressed(destination / f"world_{world_id:02d}_fits.npz",**arrays)
                    record = study_json_safe(record)
                    write_study_json(record_path,record)
                output.append(record)
                write_study_json(destination / "progress.json",{"completed_worlds":len(output),"planned_worlds":len(records)})
                progress.update(increment=1)
                print(f"Constrained fits complete for {len(output)}/{len(records)} worlds",flush=True)
        return output

    return (run_saturation_comparison,)


@app.cell
def constrained_run_control(mo):
    constrained_run_button = mo.ui.run_button(label="Run / resume constrained fits on the same 50 worlds")
    constrained_run_button
    return (constrained_run_button,)


@app.cell
def run_constrained_comparison(
    constrained_run_button,
    mo,
    run_saturation_comparison,
    saved_study_records,
):
    mo.stop(not constrained_run_button.value)
    constrained_records = run_saturation_comparison(saved_study_records,
        "notebooks/ols_study_50_seed10029", "notebooks/ols_study_50_seed10029_saturation95")
    return (constrained_records,)


@app.cell
def constrained_recovery_summary(
    StudyPath,
    constrained_records,
    mo,
    practical_summary,
    rescore_practical_recovery,
    write_study_json,
):
    constrained_scores, constrained_summary = rescore_practical_recovery(
        constrained_records,"notebooks/ols_study_50_seed10029_saturation95")
    constraint_comparison_rows = [
        {"Restriction":mode, **row}
        for mode,rows in (("Original",practical_summary),("Saturation constrained",constrained_summary))
        for row in rows if row["Fit"].startswith(("1", "2"))
    ]
    write_study_json(StudyPath("notebooks/ols_study_50_seed10029_saturation95/comparison.json"),{
        "summary":constraint_comparison_rows,
        "worlds_with_excluded_nonnull_truth":sum(r["true_nonnull_constraint_violations"]>0 for r in constrained_records),
    })
    mo.vstack([
        mo.md(f"### Original versus saturation-constrained fits\n\n"
              f"True nonzero-effect response curves violate the restriction in "
              f"**{sum(r['true_nonnull_constraint_violations']>0 for r in constrained_records)}/50 worlds**. "
              "The true-transform fits are reused unchanged."),
        mo.ui.table(constraint_comparison_rows,selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Every-channel recovery gates

    Each media/control channel gets its own allowance:
    $\|\widehat c_j-c_j\|_2 \leq \max(r\|c_j\|_2, a\|y_{noiseless}\|_2)$.
    All media must pass for media success; all controls must pass for control
    success; both groups and solver convergence are required for world success.
    The intercept is not gated. Thresholds remain 5%/10% and 0.5%/1%.

    Each channel receives the full outcome-scaled allowance; this is not a shared
    error budget. Consequently this rule can be less strict than the group gate
    when many channels have small absolute errors, or stricter when one channel
    has a large relative error. Below we compare both rules on unchanged fits.
    """)
    return


@app.cell
def channel_gate_functions(StudyPath, np, practical_group_pass):
    def channel_gate_metrics(truth, estimate, outcome_norm, relative, absolute):
        error = np.linalg.norm(estimate - truth, axis=0)
        true_norm = np.linalg.norm(truth, axis=0)
        allowance = np.maximum(relative * true_norm, absolute * outcome_norm)
        passed = np.isfinite(error) & np.isfinite(allowance) & (error <= allowance)
        ratio = np.divide(error, allowance, out=np.full_like(error, np.inf), where=allowance > 0)
        ratio = np.where((error == 0) & (allowance == 0), 0.0, ratio)
        return error, true_norm, allowance, passed, ratio


    def compare_channel_gates(records, directory, mode):
        scored = []
        for record in records:
            with np.load(StudyPath(directory)/f"world_{record['world_id']:02d}_fits.npz",allow_pickle=False) as arrays:
                truth = arrays["truth"][:,1:]
                outcome_norm = np.linalg.norm(arrays["fit_3_target"])
                n = record["media"]
                for index,fit in enumerate(record["fits"]):
                    estimate = arrays[f"fit_{index}_components"][:,1:]
                    for relative in (0.05,0.1):
                        for absolute in (0.005,0.01):
                            error,true_norm,allowance,passed,ratio = channel_gate_metrics(truth,estimate,outcome_norm,relative,absolute)
                            media_pass = bool(fit["success"] and np.all(passed[:n]))
                            control_pass = bool(fit["success"] and np.all(passed[n:]))
                            group_media = practical_group_pass(np.linalg.norm(error[:n]),np.linalg.norm(true_norm[:n]),outcome_norm,fit["success"],relative,absolute)
                            group_control = practical_group_pass(np.linalg.norm(error[n:]),np.linalg.norm(true_norm[n:]),outcome_norm,fit["success"],relative,absolute)
                            scored.append({"mode":mode,"world_id":record["world_id"],"fit":fit["label"],
                                "relative":relative,"absolute":absolute,"converged":fit["success"],
                                "media_pass":media_pass,"control_pass":control_pass,"joint_pass":media_pass and control_pass,
                                "group_joint_pass":group_media and group_control,
                                "failed_media":[name for name,ok in zip(record["terms"][1:1+n],passed[:n],strict=True) if not ok],
                                "failed_controls":[name for name,ok in zip(record["terms"][1+n:],passed[n:],strict=True) if not ok],
                                "worst_error_over_allowance":float(np.max(ratio))})
        summary = []
        for label in dict.fromkeys(row["fit"] for row in scored):
            for relative in (0.05,0.1):
                for absolute in (0.005,0.01):
                    rows = [r for r in scored if r["fit"]==label and r["relative"]==relative and r["absolute"]==absolute]
                    count = len(rows)
                    summary.append({"Restriction":mode,"Fit":label,"Relative tolerance":f"{relative:.0%}",
                        "Outcome tolerance":f"{absolute:.1%}","Worlds":count,
                        "Converged":sum(r["converged"] for r in rows),
                        "Media failure %":100*sum(not r["media_pass"] for r in rows)/count,
                        "Control failure %":100*sum(not r["control_pass"] for r in rows)/count,
                        "Channel-gated joint successes":sum(r["joint_pass"] for r in rows),
                        "Channel-gated joint failure %":100*sum(not r["joint_pass"] for r in rows)/count,
                        "Group-gated joint failure %":100*sum(not r["group_joint_pass"] for r in rows)/count,
                        "New passes":sum(r["joint_pass"] and not r["group_joint_pass"] for r in rows),
                        "Lost passes":sum(r["group_joint_pass"] and not r["joint_pass"] for r in rows)})
        return scored,summary


    # Four tiny errors pass separately, but exceed the shared group allowance.
    _gate_error, _gate_truth, _gate_allowance, _gate_pass, _gate_ratio = channel_gate_metrics(
        np.zeros((1,4)),np.full((1,4),0.4),100.0,0.05,0.005)
    assert _gate_pass.all()
    assert not practical_group_pass(np.linalg.norm(_gate_error),0.0,100.0,True,0.05,0.005)
    # One small channel's large relative error can be hidden by a large accurate channel.
    assert not channel_gate_metrics(np.array([[100.,1.]]),np.array([[100.,2.]]),10.,0.05,0.005)[3].all()
    assert practical_group_pass(1.,np.sqrt(10001),10.,True,0.05,0.005)
    assert channel_gate_metrics(np.zeros((1,1)),np.array([[0.5]]),100.,0.05,0.005)[3].all()
    assert not channel_gate_metrics(np.zeros((1,1)),np.array([[268.]]),100.,0.1,0.01)[3].all()
    return channel_gate_metrics, compare_channel_gates


@app.cell
def channel_gate_comparison(
    StudyPath,
    compare_channel_gates,
    mo,
    saved_study_records,
    study_json,
    write_study_json,
):
    mo.stop(len(list(StudyPath("notebooks/ols_study_50_seed10029_saturation95").glob("world_[0-9][0-9].json"))) != len(saved_study_records), mo.md("Constrained refits pending."))
    original_channel_scores, original_channel_summary = compare_channel_gates(
        saved_study_records,"notebooks/ols_study_50_seed10029","Original")
    constrained_channel_records = [study_json.loads(path.read_text()) for path in
        sorted(StudyPath("notebooks/ols_study_50_seed10029_saturation95").glob("world_[0-9][0-9].json"))]
    constrained_channel_scores, constrained_channel_summary = compare_channel_gates(
        constrained_channel_records,"notebooks/ols_study_50_seed10029_saturation95","Saturation constrained")
    channel_gate_summary = original_channel_summary + constrained_channel_summary
    write_study_json(StudyPath("notebooks/ols_channel_gates.json"),{
        "definition":"convergence AND every media/control channel error <= max(relative*channel truth norm,absolute*noiseless outcome norm)",
        "summary":channel_gate_summary,"scores":original_channel_scores+constrained_channel_scores})
    mo.ui.table(channel_gate_summary,selection=None)
    return (
        constrained_channel_records,
        constrained_channel_scores,
        original_channel_scores,
    )


@app.cell
def channel_gate_threshold_picker(mo):
    channel_gate_threshold = mo.ui.dropdown(options={
        "5% relative / 0.5% outcome":(0.05,0.005), "5% relative / 1% outcome":(0.05,0.01),
        "10% relative / 0.5% outcome":(0.1,0.005), "10% relative / 1% outcome":(0.1,0.01),
    },value="10% relative / 1% outcome",label="Per-channel gate thresholds")
    channel_gate_threshold
    return (channel_gate_threshold,)


@app.cell
def selected_channel_gates(
    channel_gate_metrics,
    channel_gate_threshold,
    fit_restriction_picker,
    mo,
    np,
    study_fit_picker,
    viewed_study_arrays,
    viewed_study_fit,
    viewed_study_record,
):
    gate_error, gate_truth_norm, gate_allowance, gate_pass, gate_ratio = channel_gate_metrics(
        viewed_study_arrays["truth"][:,1:],
        viewed_study_arrays[f"fit_{study_fit_picker.value}_components"][:,1:],
        np.linalg.norm(viewed_study_arrays["fit_3_target"]), *channel_gate_threshold.value,
    )
    channel_detail_rows = [
        {"Channel":name,"Group":"Media" if index<viewed_study_record["media"] else "Control",
         "Error norm":float(gate_error[index]),"True norm":float(gate_truth_norm[index]),
         "Allowed error":float(gate_allowance[index]),"Error / allowance":float(gate_ratio[index]),
         "Error gate passes":bool(gate_pass[index])}
        for index,name in enumerate(viewed_study_record["terms"][1:])
    ]
    mo.vstack([
        mo.md(f"### Per-channel gates: study world {viewed_study_record['world_id']+1}\n\n"
              f"{fit_restriction_picker.value} · {viewed_study_fit['label']}\n\n"
              f"**Converged:** {viewed_study_fit['success']} · "
              f"**Failing error gates:** {np.count_nonzero(~gate_pass)} / {len(gate_pass)} · "
              f"**World passes:** {bool(viewed_study_fit['success'] and gate_pass.all())}"),
        mo.ui.table(channel_detail_rows,selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 50-world oracle-graph MMM: generator continuous priors

    All methods know the true zero/nonzero media and control identities. OLS omits known-null predictors and reports their contributions as exactly zero. Bayesian inference additionally conditions on the generator's parameter-only saturation anchors so its continuous priors have the correct physical meaning. This is an **oracle-graph, fixed-anchor benchmark**, not variable selection or unprivileged inference.

    Generator priors: baseline Uniform(3,8); decay Uniform(.2,.8); log MM multiplier Uniform(log(.05),log(20)); media reference contribution Uniform(.5,2); control reference contribution Uniform(-.3,.3), divided by reference scale 2; relative noise Uniform(.01,.028), multiplied by the norm of effective media coefficients. Null coefficients are exactly zero. No realized nonzero coefficient, shape, baseline or noise value initializes inference. Known-null shape parameters in OLS are placeholders, not estimates.

    NUTS uses Nutpie draw_diag, four chains, 1000 tune + 1000 draws, target_accept .95. All 50 saved worlds use the same model and recovery thresholds; the completed first five fits are reused. Diagnostics exclude constant known-null contribution terms. Posterior intervals and point recovery remain separate; coverage over all active inputs includes trivially covered known zeros.
    """)
    return


@app.cell
def bayesian_imports():
    import pymc as bayes_pm
    import nutpie as bayes_nutpie
    import arviz as bayes_az
    import xarray as bayes_xr

    return bayes_az, bayes_nutpie, bayes_pm, bayes_xr


@app.cell
def bayesian_model_definition(
    SlotLayout,
    StudyPath,
    bayes_pm,
    constant_baseline_cfg,
    mechanisms,
    mo,
    np,
    pg,
    pt,
    saved_study_records,
):
    def bayesian_world_data(record):
        corpus = pg.load_corpus(StudyPath("notebooks/ols_study_50_seed10029") / f"batch_{record['batch']:02d}.npz")
        world = record["local_world"]
        media = corpus["treatment_raw"][world].astype(np.float64)
        controls = corpus["covariates"][world].astype(np.float64)
        media_scale = np.array([np.mean(x[x > 0]) if np.any(x > 0) else 1.0 for x in media.T])
        control_scale = np.sqrt(np.mean(controls ** 2, axis=0))
        control_scale = np.where(control_scale > 0, control_scale, 1.0)
        outcome = corpus["outcome_raw"][world, 7:].astype(np.float64)
        outcome_scale = np.array(max(float(outcome.std()), 1e-8))
        graph = SlotLayout(15,10,1).unpack(corpus["g"][world])
        active_media = corpus["treatment_active_mask"][world].astype(np.float64)
        anchor = np.where(active_media,corpus["mechanism_saturation_scale"][world],1.0).astype(np.float64)
        # Oracle information is restricted to graph identities and the physical reference anchor.
        # No realized coefficients, transformations, intercept, or noise scales initialize the fit.
        data = {"media_data":media/media_scale,"control_data":controls/control_scale,
            "media_scale":media_scale,"control_scale":control_scale,
            "media_mask":active_media,"control_mask":corpus["covariate_active_mask"][world].astype(np.float64),
            "media_effect_mask":graph["cy"].astype(np.float64),"control_effect_mask":graph["zy"].astype(np.float64),
            "saturation_anchor":anchor,"outcome_scale":outcome_scale,"outcome_data":outcome/outcome_scale}
        assert all(np.isfinite(value).all() for value in data.values())
        assert np.all(anchor>0) and data["media_effect_mask"].sum()>=1
        return data, corpus


    def build_bayesian_mmm(data):
        cfg=constant_baseline_cfg
        coords = {"week":np.arange(104),"fit_week":np.arange(7,104),"media":np.arange(15),"control":np.arange(10)}
        assert cfg.mm_scale_prior=="log_uniform" and cfg.treatment_reference_multiplier==1.0
        with bayes_pm.Model(coords=coords) as model:
            x=bayes_pm.Data("media_data",data["media_data"],dims=("week","media"))
            z=bayes_pm.Data("control_data",data["control_data"],dims=("week","control"))
            media_scale=bayes_pm.Data("media_scale",data["media_scale"],dims="media")
            control_scale=bayes_pm.Data("control_scale",data["control_scale"],dims="control")
            media_mask=bayes_pm.Data("media_mask",data["media_mask"],dims="media")
            control_mask=bayes_pm.Data("control_mask",data["control_mask"],dims="control")
            media_effect_mask=bayes_pm.Data("media_effect_mask",data["media_effect_mask"],dims="media")
            control_effect_mask=bayes_pm.Data("control_effect_mask",data["control_effect_mask"],dims="control")
            anchor=bayes_pm.Data("saturation_anchor",data["saturation_anchor"],dims="media")
            y_scale=bayes_pm.Data("outcome_scale",data["outcome_scale"])
            y=bayes_pm.Data("outcome_data",data["outcome_data"],dims="fit_week")
            intercept=bayes_pm.Uniform("intercept",*cfg.rw_baseline_mean_range)
            alpha=bayes_pm.Uniform("alpha",*cfg.carryover_alpha_range,dims="media")
            k_lo,k_hi=cfg.saturation_prior_ranges["michaelis_menten"]["kappa_mult"]
            log_kappa=bayes_pm.Uniform("log_kappa",np.log(k_lo),np.log(k_hi),dims="media")
            kappa=bayes_pm.Deterministic("mm_kappa_mult",pt.clip(pt.exp(log_kappa),k_lo,k_hi),dims="media")
            half=bayes_pm.Deterministic("half_saturation",kappa*anchor,dims="media")
            half_ratio=bayes_pm.Deterministic("half_ratio",half/media_scale,dims="media")
            amplitude=bayes_pm.Uniform("amplitude",*cfg.treatment_reference_contribution_range,dims="media")
            # Effective coefficients include the known graph, so all null contributions are exactly zero.
            beta=bayes_pm.Deterministic("beta",amplitude*(1+kappa)*media_effect_mask,dims="media")
            control_reference=bayes_pm.Uniform("control_reference_contribution",*cfg.covariate_reference_contribution_range,dims="control")
            rho=bayes_pm.Deterministic("rho",control_reference/cfg.covariate_reference_scale*control_effect_mask,dims="control")
            control_amplitude=bayes_pm.Deterministic("control_amplitude",rho*control_scale,dims="control")
            responses=pt.stack([
                mechanisms.stable_michaelis_menten_kappa_relative(
                    mechanisms.apply_geometric_carryover(x[:,channel:channel+1],alpha[channel],8)[:,0],
                    1.0,kappa_mult=half_ratio[channel]) for channel in range(15)
            ],axis=1)[7:]
            media_effect=responses*beta*media_mask
            control_effect=z[7:]*control_amplitude*control_mask
            bayes_pm.Deterministic("mean_media_contribution",media_effect.mean(axis=0),dims="media")
            bayes_pm.Deterministic("mean_control_contribution",control_effect.mean(axis=0),dims="control")
            mu=bayes_pm.Deterministic("mu",intercept+media_effect.sum(axis=1)+control_effect.sum(axis=1),dims="fit_week")
            sigma_relative=bayes_pm.Uniform("sigma_relative",*cfg.rw_outcome_std_range)
            sigma=bayes_pm.Deterministic("sigma",sigma_relative*pt.sqrt(pt.sum(beta**2)))
            bayes_pm.Normal("outcome",mu=mu/y_scale,sigma=sigma/y_scale,observed=y,dims="fit_week")
        return model


    bayesian_initial_data,bayesian_initial_corpus=bayesian_world_data(saved_study_records[0])
    bayesian_model=build_bayesian_mmm(bayesian_initial_data)
    bayesian_initial_logp=bayesian_model.compile_logp()(bayesian_model.initial_point())
    bayesian_initial_gradient=bayesian_model.compile_dlogp()(bayesian_model.initial_point())
    assert np.isfinite(bayesian_initial_logp) and np.isfinite(bayesian_initial_gradient).all()
    mo.md(f"Generator-prior oracle model built; initial log density {bayesian_initial_logp:.2f}; gradients finite. Known graph and generator reference anchors are supplied; effect sizes and shapes remain unknown.")

    return (
        bayesian_initial_corpus,
        bayesian_model,
        bayesian_world_data,
        build_bayesian_mmm,
    )


@app.cell
def bayesian_prior_check(
    bayes_pm,
    bayesian_initial_corpus,
    bayesian_model,
    mo,
    np,
):
    with bayesian_model:
        bayesian_prior = bayes_pm.sample_prior_predictive(draws=32,random_seed=71000)
    bayesian_prior_mu = bayesian_prior["prior"]["mu"].values
    bayesian_prior_report = {
        "prior_mu_5_50_95":np.quantile(bayesian_prior_mu,[0.05,0.5,0.95]).tolist(),
        "observed_outcome_min_max":[float(bayesian_initial_corpus["outcome_raw"][0].min()),float(bayesian_initial_corpus["outcome_raw"][0].max())],
        "prior_predictive_negative_fraction":float(np.mean(bayesian_prior["prior_predictive"]["outcome"].values < 0)),
    }
    mo.md(f"Prior predictive check (32 draws): {bayesian_prior_report}. Gaussian observation noise and signed controls can give negative outcomes even with nonnegative media effects; no tuning to the realized world is performed.")
    return bayesian_prior, bayesian_prior_report


@app.cell
def bayesian_nutpie_compilation(
    bayes_nutpie,
    bayesian_model,
    mo,
    perf_counter,
):
    bayesian_compile_started = perf_counter()
    import pandas as bayes_pd
    # Nutpie 0.16.11 expects NumPy-backed string coordinates, not Pandas StringArray.
    with bayes_pd.option_context("future.infer_string", False):
        bayesian_compiled = bayes_nutpie.compile_pymc_model(bayesian_model,backend="numba",gradient_backend="pytensor",freeze_model=False)
    bayesian_compile_seconds = perf_counter()-bayesian_compile_started
    mo.md(f"Nutpie model compiled once in **{bayesian_compile_seconds:.1f}s**. Sampling will use adaptation='draw_diag'.")
    return bayes_pd, bayesian_compile_seconds, bayesian_compiled


@app.cell
def bayesian_study_functions(
    StudyPath,
    bayes_az,
    bayes_nutpie,
    bayes_pm,
    bayes_xr,
    bayesian_compile_seconds,
    bayesian_compiled,
    bayesian_prior,
    bayesian_prior_report,
    bayesian_world_data,
    build_bayesian_mmm,
    channel_gate_metrics,
    constant_baseline_cfg,
    mo,
    np,
    perf_counter,
    prepare_ols_case,
    relative_component_error,
    replace,
    saved_study_records,
    study_hashlib,
    study_inspect,
    study_json,
    study_json_safe,
    write_study_json,
):
    def posterior_component_draws(idata, data, record):
        posterior = idata["posterior"]
        alpha = posterior["alpha"].transpose("chain","draw","media").values.reshape(-1,15)
        half = posterior["half_ratio"].transpose("chain","draw","media").values.reshape(-1,15)
        beta = posterior["beta"].transpose("chain","draw","media").values.reshape(-1,15)
        rho = posterior["rho"].transpose("chain","draw","control").values.reshape(-1,10)
        intercept = posterior["intercept"].values.reshape(-1)
        weights = alpha[:,None,:] ** np.arange(8)[None,:,None]
        weights /= weights.sum(axis=1,keepdims=True)
        history = np.zeros((104,8,15))
        for lag in range(8):
            history[lag:,lag,:] = data["media_data"][:104-lag]
        carried = np.einsum("slk,tlk->stk",weights,history)
        media = carried/(carried+half[:,None,:])*beta[:,None,:]
        controls = data["control_data"][7:]*data["control_scale"]
        media_slots = np.flatnonzero(data["media_mask"])
        control_slots = np.flatnonzero(data["control_mask"])
        components = np.concatenate([
            np.broadcast_to(intercept[:,None,None],(len(intercept),97,1)),
            media[:,7:,media_slots],
            controls[None,:,control_slots]*rho[:,None,control_slots],
        ],axis=2)
        # Independent NumPy reconstruction must agree with PyMC's symbolic mu.
        mu = posterior["mu"].transpose("chain","draw","fit_week").values.reshape(-1,97)
        np.testing.assert_allclose(components.sum(axis=2),mu,rtol=1e-10,atol=1e-10)
        return components,mu


    def diagnose_bayesian_world(idata,data,corpus,record,directory):
        media_slots = np.flatnonzero(data["media_effect_mask"])
        control_slots = np.flatnonzero(data["control_effect_mask"])
        variables = ["intercept","sigma","sigma_relative","alpha","mm_kappa_mult","half_ratio","amplitude","mean_media_contribution"]
        if len(control_slots):
            variables += ["control_reference_contribution","control_amplitude","mean_control_contribution"]
        diagnostic = bayes_az.summary(idata,var_names=variables,coords={"media":media_slots,**({"control":control_slots} if len(control_slots) else {})},
                                      ci_prob=0.95,ci_kind="eti",round_to="none")
        diagnostic.to_csv(directory/f"world_{record['world_id']:02d}_diagnostics.csv")
        stats = idata["sample_stats"]
        divergences = int(stats["diverging"].sum())
        depth_name = next((name for name in ("depth","tree_depth") if name in stats),None)
        draws_at_depth_limit = int((stats[depth_name]>=10).sum()) if depth_name else None
        # Reaching depth 10 is not itself a truncated trajectory; use Nutpie's flag.
        depth_hits = int(stats["maxdepth_reached"].sum()) if "maxdepth_reached" in stats else None
        energy_name = "energy" if "energy" in stats else None
        bfmi = (np.mean(np.diff(stats[energy_name].values,axis=1)**2,axis=1)/np.var(stats[energy_name].values,axis=1,ddof=1)) if energy_name else np.array([np.nan])
        rhats = diagnostic["r_hat"].to_numpy()
        bulk = diagnostic["ess_bulk"].to_numpy(); tail = diagnostic["ess_tail"].to_numpy()
        healthy = bool(divergences==0 and depth_hits==0 and np.isfinite(rhats).all() and np.max(rhats)<1.01
                       and np.isfinite(bulk).all() and np.min(bulk)>=400 and np.isfinite(tail).all() and np.min(tail)>=400
                       and np.isfinite(bfmi).all() and np.min(bfmi)>=0.3)
        cfg = replace(constant_baseline_cfg,seed=record["seed"])
        case = prepare_ols_case(corpus,record["local_world"],cfg)
        components,mu = posterior_component_draws(idata,data,record)
        point = components.mean(axis=0)
        low,high = np.quantile(components,[0.025,0.975],axis=0)
        n = record["media"]
        # Descriptive posterior uncertainty at the SAME channel-wise recovery tolerances.
        # Uses generator truth only for evaluation, never for fitting.
        draw_errors=np.linalg.norm(components[:,:,1:]-case["truth"][None,:,1:],axis=1)
        posterior_radius95=np.quantile(np.linalg.norm(components[:,:,1:]-point[None,:,1:],axis=1),0.95,axis=0)
        nonnull=np.r_[data["media_effect_mask"][data["media_mask"].astype(bool)],
                      data["control_effect_mask"][data["control_mask"].astype(bool)]].astype(bool)
        gates=[]
        for relative in (0.05,0.1):
            for absolute in (0.005,0.01):
                error,truth_norm,allowed,passed,ratio = channel_gate_metrics(case["truth"][:,1:],point[:,1:],
                    np.linalg.norm(case["noiseless"][7:]),relative,absolute)
                gates.append({"relative":relative,"absolute":absolute,"error_gates_pass":bool(passed.all()),
                    "diagnostics_pass":healthy,"world_pass":bool(healthy and passed.all()),
                    "posterior_mass_within_all_channel_tolerances":float(np.mean(np.all(draw_errors<=allowed,axis=1))),
                    "max_nonnull_radius95_over_allowance":float(np.max(posterior_radius95[nonnull]/allowed[nonnull])),
                    "median_nonnull_radius95_over_allowance":float(np.median(posterior_radius95[nonnull]/allowed[nonnull])),
                    "media_pass":bool(healthy and passed[:n].all()),"control_pass":bool(healthy and passed[n:].all()),
                    "failing_channels":[name for name,ok in zip(case["names"][1:],passed,strict=True) if not ok]})
        posterior = idata["posterior"]
        sigma = posterior["sigma"].values.reshape(-1)
        rng=np.random.default_rng(72001+record["world_id"])
        replicated=mu+rng.normal(size=mu.shape)*sigma[:,None]
        pp_low,pp_high=np.quantile(replicated,[0.025,0.975],axis=0)
        outcome=case["observed"][7:]
        summary={"world_id":record["world_id"],"diagnostics_pass":healthy,"divergences":divergences,
            "max_rhat":float(np.max(rhats)),"min_bulk_ess":float(np.min(bulk)),"min_tail_ess":float(np.min(tail)),
            "maxdepth_hits":depth_hits,"draws_at_depth_limit":draws_at_depth_limit,"min_bfmi":float(np.min(bfmi)),
            "max_mean_mcse":float(diagnostic["mcse_mean"].max()),
            "media_relative_error":relative_component_error(point[:,1:1+n],case["truth"][:,1:1+n]),
            "control_relative_error":relative_component_error(point[:,1+n:],case["truth"][:,1+n:]),
            "outcome_rmse":float(np.sqrt(np.mean((point.sum(axis=1)-outcome)**2))),
            "noiseless_rmse":float(np.sqrt(np.mean((point.sum(axis=1)-case["noiseless"][7:])**2))),
            "media_95pct_pointwise_truth_coverage":float(np.mean((case["truth"][:,1:1+n]>=low[:,1:1+n])&(case["truth"][:,1:1+n]<=high[:,1:1+n]))),
            "control_95pct_pointwise_truth_coverage":float(np.mean((case["truth"][:,1+n:]>=low[:,1+n:])&(case["truth"][:,1+n:]<=high[:,1+n:]))),
            "observed_95pct_predictive_coverage":float(np.mean((outcome>=pp_low)&(outcome<=pp_high))),
            "gates":gates,"terms":case["names"],
            "sample_stats_variables":list(stats.data_vars)}
        np.savez_compressed(directory/f"world_{record['world_id']:02d}_recovery.npz",truth=case["truth"],mean=point,
            lower95=low,upper95=high,observed=outcome,noiseless=case["noiseless"][7:],
            predictive_lower95=pp_low,predictive_upper95=pp_high)
        return study_json_safe(summary)


    def run_bayesian_worlds():
        directory=StudyPath("notebooks/bayesian_mmm_draw_diag")
        source=StudyPath("notebooks/ols_study_50_seed10029")
        records=saved_study_records[:50]
        assert [r["world_id"] for r in records]==list(range(50))
        model_source=study_inspect.getsource(build_bayesian_mmm)+study_inspect.getsource(bayesian_world_data)
        manifest={"world_ids":list(range(50)),"sampler":"nutpie","adaptation":"draw_diag", "backend":"numba","oracle_graph":True,"known_reference_anchors":True,
            "chains":4,"tune":1000,"draws":1000,"target_accept":0.95,"maxdepth":10,
            "seeds":list(range(71001,71051)),"pymc":bayes_pm.__version__,"nutpie":bayes_nutpie.__version__,
            "model_sha256":study_hashlib.sha256(model_source.encode()).hexdigest(),
            "source_manifest":study_json.loads((source/"manifest.json").read_text())}
        directory.mkdir(parents=True,exist_ok=True)
        if (directory/"manifest.json").exists():
            assert study_json.loads((directory/"manifest.json").read_text())==manifest,"MCMC checkpoint manifest differs"
        else:
            if any(directory.iterdir()):raise ValueError("Refusing a nonempty MCMC directory without manifest")
            write_study_json(directory/"manifest.json",manifest)
            (directory/"model.py.txt").write_text(model_source)
            write_study_json(directory/"prior_check.json",bayesian_prior_report)
            bayesian_prior.to_netcdf(directory/"prior_predictive.nc",engine="h5netcdf")
        reports=[]
        for record in records:
            world_id=record["world_id"]
            mo.output.replace(mo.md(f"### NUTS: {len(reports)}/50 completed · processing world {world_id+1}\n\nEach posterior is saved before diagnosis. Existing fits are reused."))
            data,corpus=bayesian_world_data(record)
            posterior_path=directory/f"world_{world_id:02d}_posterior.nc"
            timing_path=directory/f"world_{world_id:02d}_timing.json"
            if posterior_path.exists():
                idata=bayes_az.from_netcdf(posterior_path,engine="h5netcdf")
                timing=study_json.loads(timing_path.read_text())
            else:
                bound=bayesian_compiled.with_data(**data)
                started=perf_counter()
                idata=bayes_nutpie.sample(bound,chains=4,cores=4,tune=1000,draws=1000,
                    seed=71001+world_id,adaptation="draw_diag",target_accept=0.95,maxdepth=10,
                    save_warmup=True,progress_bar=False)
                timing={"sampling_seconds":perf_counter()-started,"shared_compilation_seconds":bayesian_compile_seconds}
                # Explicit per-world provenance; do not retain first-world data after rebinding.
                idata["observed_data"]=bayes_xr.Dataset({"outcome_scaled":("fit_week",data["outcome_data"]),
                    "outcome_raw":("fit_week",data["outcome_data"]*data["outcome_scale"])},coords={"fit_week":np.arange(7,104)})
                idata["constant_data"]=bayes_xr.Dataset({
                    "media_data":(("week","media"),data["media_data"]),
                    "control_data":(("week","control"),data["control_data"]),
                    "media_scale":("media",data["media_scale"]),"control_scale":("control",data["control_scale"]),
                    "media_mask":("media",data["media_mask"]),"control_mask":("control",data["control_mask"]),
                    "media_effect_mask":("media",data["media_effect_mask"]),
                    "control_effect_mask":("control",data["control_effect_mask"]),
                    "saturation_anchor":("media",data["saturation_anchor"]),
                    "outcome_scale":data["outcome_scale"]},
                    coords={"week":np.arange(104),"media":np.arange(15),"control":np.arange(10)})
                idata.attrs.update(study_world_id=world_id,generation_seed=record["seed"],sampler="nutpie",adaptation="draw_diag")
                write_study_json(timing_path,timing)
                idata.to_netcdf(posterior_path,engine="h5netcdf")
            # Posterior is safely saved before diagnostics or recovery calculations.
            report=diagnose_bayesian_world(idata,data,corpus,record,directory)
            report.update(timing)
            write_study_json(directory/f"world_{world_id:02d}_report.json",report)
            reports.append(report)
            write_study_json(directory/"reports.json",reports)
            write_study_json(directory/"progress.json",{"completed_worlds":len(reports),"planned_worlds":50,
                "complete":len(reports)==50,"diagnostics_pass":sum(r["diagnostics_pass"] for r in reports),
                "sampling_seconds":sum(r["sampling_seconds"] for r in reports)})
            print(f"MCMC world {world_id+1}/50: divergences={report['divergences']}, Rhat={report['max_rhat']:.4f}, minESS={report['min_bulk_ess']:.0f}, diagnostics_pass={report['diagnostics_pass']}",flush=True)
        return reports

    return posterior_component_draws, run_bayesian_worlds


@app.cell
def bayesian_run_control(mo):
    bayesian_run_button=mo.ui.run_button(label="Run / resume Nutpie draw_diag on all 50 worlds")
    bayesian_run_button
    return (bayesian_run_button,)


@app.cell
def run_bayesian_first_five(bayesian_run_button, mo, run_bayesian_worlds):
    mo.stop(not bayesian_run_button.value)
    bayesian_reports=run_bayesian_worlds()
    return (bayesian_reports,)


@app.cell
def bayesian_first_five_report(bayesian_reports, mo):
    bayesian_summary_rows=[{
        "Study world":r["world_id"]+1,"Diagnostics pass":r["diagnostics_pass"],"Divergences":r["divergences"],
        "Max-depth hits":r["maxdepth_hits"],"Min BFMI":r["min_bfmi"],"Max Rhat":r["max_rhat"],"Min bulk ESS":r["min_bulk_ess"],"Min tail ESS":r["min_tail_ess"],
        "Media relative error":r["media_relative_error"],"Control relative error":r["control_relative_error"],
        "Noiseless outcome RMSE":r["noiseless_rmse"],"Media 95% coverage":r["media_95pct_pointwise_truth_coverage"],
        "Control 95% coverage":r["control_95pct_pointwise_truth_coverage"],"Sampling seconds":r["sampling_seconds"],
        "Pass at 10% / 1%":r["gates"][-1]["world_pass"],
    } for r in bayesian_reports]
    mo.vstack([mo.md("### 50-world MCMC results (Nutpie draw_diag)\n\nCoverage is pointwise across contributions/weeks, not a simultaneous or population calibration guarantee."),
               mo.ui.table(bayesian_summary_rows,selection=None)])
    return


@app.cell
def bayesian_paired_comparison_report(
    StudyPath,
    bayesian_reports,
    constrained_channel_records,
    mo,
    saved_study_records,
    write_study_json,
):
    bayesian_paired_comparison = [
        {"Study world":report["world_id"]+1,
         "Original noisy OLS media error":saved_study_records[report["world_id"]]["fits"][0]["media_error"],
         "Constrained noisy OLS media error":constrained_channel_records[report["world_id"]]["fits"][0]["media_error"],
         "True-transform noisy OLS media error":saved_study_records[report["world_id"]]["fits"][2]["media_error"],
         "Bayesian mean media error":report["media_relative_error"],
         "Original noisy OLS control error":saved_study_records[report["world_id"]]["fits"][0]["control_error"],
         "Constrained noisy OLS control error":constrained_channel_records[report["world_id"]]["fits"][0]["control_error"],
         "True-transform noisy OLS control error":saved_study_records[report["world_id"]]["fits"][2]["control_error"],
         "Bayesian mean control error":report["control_relative_error"],
         "MCMC diagnostics pass":report["diagnostics_pass"]}
        for report in bayesian_reports
    ]
    write_study_json(StudyPath("notebooks/bayesian_mmm_draw_diag/paired_comparison.json"),bayesian_paired_comparison)
    mo.vstack([mo.md("### Paired noisy-world comparison\n\nErrors below are group relative L2, not percentages. Bayesian priors differ from OLS; these are not sampler-only comparisons. The true-transform OLS column has additional privileged information."),
               mo.ui.table(bayesian_paired_comparison,selection=None)])
    return


@app.cell
def bayesian_world_selector(bayesian_reports, mo):
    bayesian_world_picker=mo.ui.dropdown(options={f"Study world {r['world_id']+1}":r["world_id"] for r in bayesian_reports},
        value="Study world 3",label="MCMC world")
    bayesian_world_picker
    return (bayesian_world_picker,)


@app.cell(hide_code=True)
def scorecard_geometry_functions(np):
    def contribution_geometry(truth, names, coefficients):
        # Use known nonzero coefficients, not fitted magnitude or an arbitrary threshold.
        indices=[i for i,name in enumerate(names) if name!="Intercept" and coefficients[i]!=0]
        series=np.asarray(truth,dtype=float)[:,indices]
        means=series.mean(axis=0)
        sd=series.std(axis=0)
        constant=np.ptp(series,axis=0)==0
        sd[constant]=0.0
        means[constant]=series[0,constant]
        cv=np.divide(sd,np.abs(means),out=np.full(len(indices),np.nan),where=means!=0)
        # Constants have undefined Pearson correlations, including their diagonal.
        corr=np.full((len(indices),len(indices)),np.nan)
        variable=np.flatnonzero(~constant)
        if len(variable):
            centered=series[:,variable]-means[variable]
            unit=centered/np.linalg.norm(centered,axis=0)
            corr[np.ix_(variable,variable)]=np.clip(unit.T@unit,-1,1)
        rows=[{"Channel":names[i],"Kind":"Media" if names[i].startswith("C") else "Control",
               "Mean":float(means[j]),"SD":float(sd[j]),"CV":float(cv[j]) if np.isfinite(cv[j]) else None,
               "Constant":bool(constant[j])} for j,i in enumerate(indices)]
        return {"indices":indices,"rows":rows,"correlation":corr}


    def flatness_status(row, cutoff):
        if row["Constant"]:
            return "Constant"
        if row["CV"] is None:
            return "CV undefined (zero mean)"
        return "Near-flat" if row["CV"]<=cutoff else "Variable"


    return contribution_geometry, flatness_status


@app.cell(hide_code=True)
def scorecard_geometry_cache(
    StudyPath,
    contribution_geometry,
    np,
    saved_study_records,
):
    scorecard_geometry_by_world={}
    for _record in saved_study_records:
        with np.load(StudyPath("notebooks/ols_study_50_seed10029")/f"world_{_record['world_id']:02d}_fits.npz",allow_pickle=False) as _a:
            scorecard_geometry_by_world[_record["world_id"]]=contribution_geometry(_a["truth"],_record["terms"],_a["true_coefficients"])

    return (scorecard_geometry_by_world,)


@app.cell(hide_code=True)
def generator_bounded_ols_functions(
    least_squares,
    np,
    perf_counter,
    transform_media_columns,
):
    from scipy.optimize import lsq_linear as bounded_lsq_linear


    def generator_box_spec(cfg):
        assert cfg.mm_scale_prior=="log_uniform" and cfg.treatment_reference_multiplier==1.0
        return {"intercept":list(cfg.rw_baseline_mean_range),"alpha":list(cfg.carryover_alpha_range),
            "kappa":list(cfg.saturation_prior_ranges["michaelis_menten"]["kappa_mult"]),
            "amplitude":list(cfg.treatment_reference_contribution_range),
            "control_reference":list(cfg.covariate_reference_contribution_range),
            "control_reference_scale":float(cfg.covariate_reference_scale)}


    def solve_reference_box(design,target,lower,upper):
        scale=np.sqrt(np.mean(design**2,axis=0))
        scale=np.where(scale>0,scale,1.0)
        result=bounded_lsq_linear(design/scale,target,bounds=(lower*scale,upper*scale),method="bvls",tol=1e-10,max_iter=1000)
        return result.x/scale,result


    def fit_generator_bounded(media,controls,anchor,target,box,*,fixed=None,seed=2026,starts=2,max_nfev=120):
        # target is already trimmed; full media history is retained for transformation.
        began=perf_counter();m=media.shape[1];z=controls.shape[1];calls=0
        lower=np.r_[box["intercept"][0],np.full(m,box["amplitude"][0]),np.full(z,box["control_reference"][0])]
        upper=np.r_[box["intercept"][1],np.full(m,box["amplitude"][1]),np.full(z,box["control_reference"][1])]
        def design_at(alpha,kappa):
            response=transform_media_columns(media,alpha,kappa*anchor)[7:]
            return np.column_stack([np.ones(len(target)),response*(1+kappa),controls[7:]/box["control_reference_scale"]])
        attempts=[]
        if fixed is not None:
            alpha,kappa=map(np.asarray,fixed)
            design=design_at(alpha,kappa)
            weights,result=solve_reference_box(design,target,lower,upper)
            success=bool(result.success);message=str(result.message);nfev=int(result.nit)
            optimality=float(result.optimality);active=int(np.count_nonzero(result.active_mask))
        else:
            weight_count=len(lower)
            lo=np.r_[lower,np.full(m,box["alpha"][0]),np.full(m,np.log(box["kappa"][0]))]
            hi=np.r_[upper,np.full(m,box["alpha"][1]),np.full(m,np.log(box["kappa"][1]))]
            target_scale=max(float(np.std(target)),1e-8)
            def unpack(theta):
                return theta[:weight_count],theta[weight_count:weight_count+m],np.exp(theta[weight_count+m:])
            def residual(theta):
                nonlocal calls
                calls+=1
                w,a,k=unpack(theta)
                return (design_at(a,k)@w-target)/target_scale
            rng=np.random.default_rng(seed);candidates=[]
            for start in range(starts):
                a=np.full(m,np.mean(box["alpha"])) if start==0 else rng.uniform(*box["alpha"],size=m)
                lk=np.full(m,np.mean(np.log(box["kappa"]))) if start==0 else rng.uniform(*np.log(box["kappa"]),size=m)
                w,_=solve_reference_box(design_at(a,np.exp(lk)),target,lower,upper)
                initial=np.r_[w,a,lk]
                initial=np.clip(initial,lo+1e-9*(hi-lo),hi-1e-9*(hi-lo))
                candidate=least_squares(residual,initial,bounds=(lo,hi),x_scale=hi-lo,max_nfev=max_nfev,ftol=1e-8,xtol=1e-8,gtol=1e-8)
                candidates.append(candidate)
                attempts.append({"start":start+1,"success":bool(candidate.success),"nfev":int(candidate.nfev),"cost":float(candidate.cost),"message":str(candidate.message)})
            result=min(candidates,key=lambda c:c.cost)
            weights,alpha,kappa=unpack(result.x)
            design=design_at(alpha,kappa)
            success=bool(result.success);message=str(result.message);nfev=sum(r["nfev"] for r in attempts)
            optimality=float(result.optimality);active=int(np.count_nonzero(result.active_mask))
        components=design*weights
        coefficients=np.r_[weights[0],weights[1:1+m]*(1+kappa),weights[1+m:]/box["control_reference_scale"]]
        arrays={"reference_weights":weights,"alpha":alpha,"kappa":kappa,"half":kappa*anchor,"coefficients":coefficients,"components":components,"prediction":components.sum(axis=1)}
        report={"success":success,"message":message,"nfev":nfev,"residual_calls":calls,"optimality":optimality,"boundary_count":active,"attempts":attempts,"seconds":perf_counter()-began,
            "observed_rmse":float(np.sqrt(np.mean((arrays["prediction"]-target)**2)))}
        return arrays,report


    def score_bounded_components(components,truth,names,success):
        gates=[]
        for relative in (.05,.1):
            for absolute in (.005,.01):
                allowance=np.maximum(relative*np.linalg.norm(truth,axis=0),absolute*np.linalg.norm(truth.sum(axis=1)))
                error=np.linalg.norm(components-truth,axis=0)
                failures=[names[i] for i in range(1,len(names)) if error[i]>allowance[i]]
                gates.append({"relative":relative,"absolute":absolute,"error_gates_pass":not failures,"world_pass":bool(success and not failures),"failing_channels":failures})
        return gates


    return (
        fit_generator_bounded,
        generator_box_spec,
        score_bounded_components,
        solve_reference_box,
    )


@app.cell(hide_code=True)
def generator_bounded_ols_runner(
    StudyPath,
    bayesian_world_data,
    constant_baseline_cfg,
    fit_generator_bounded,
    generator_box_spec,
    np,
    saved_study_records,
    score_bounded_components,
    solve_reference_box,
    study_hashlib,
    study_inspect,
    study_json,
    write_study_json,
):
    def run_generator_bounded_study():
        import ast
        import textwrap
        directory=StudyPath("notebooks/ols_study_50_seed10029_generator_bounds")
        directory.mkdir(exist_ok=True)
        box=generator_box_spec(constant_baseline_cfg)
        assert constant_baseline_cfg.l_max==8 and constant_baseline_cfg.carryover_burn_in
        source="\n".join(ast.dump(ast.parse(textwrap.dedent(study_inspect.getsource(f))),include_attributes=False) for f in (generator_box_spec,solve_reference_box,fit_generator_bounded,score_bounded_components))
        provenance={"bounds":box,"source_sha256":study_hashlib.sha256(source.encode()).hexdigest(),
            "data_sha256":{str(p.name):study_hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(StudyPath("notebooks/ols_study_50_seed10029").glob("batch_*.npz"))},
            "starts":2,"max_nfev_per_start":120,"initialization":"midpoint shapes + seeded random shapes; data-only bounded-linear weights", "seed":2026,
            "objective":"observed-outcome unweighted squared residuals, scaled by outcome SD; not Bayesian likelihood or MAP",
            "oracle_information":"graph and generator reference anchors; true alpha/kappa only for fixed-transform variant", "fit_weeks":[7,103]}
        manifest=directory/"manifest.json"
        if manifest.exists():
            if study_json.loads(manifest.read_text())!=provenance:
                raise ValueError("Bounded OLS cache provenance changed; explicit invalidation is required before refitting")
        else:
            write_study_json(manifest,provenance)
        reports=[]
        for record in saved_study_records:
            world=record["world_id"];report_path=directory/f"world_{world:02d}.json";array_path=directory/f"world_{world:02d}_fits.npz"
            if report_path.exists() and array_path.exists():
                reports.append(study_json.loads(report_path.read_text()));continue
            data,corpus=bayesian_world_data(record)
            w=record["local_world"]
            ms=np.flatnonzero(data["media_effect_mask"]);cs=np.flatnonzero(data["control_effect_mask"])
            x=corpus["treatment_raw"][w][:,ms].astype(float);z=corpus["covariates"][w][:,cs].astype(float)
            target=corpus["outcome_raw"][w,7:].astype(float)
            names=record["terms"]
            positions=[0]+[names.index(f"C{i+1}") for i in ms]+[names.index(f"Z{i+1}") for i in cs]
            with np.load(StudyPath("notebooks/ols_study_50_seed10029")/f"world_{world:02d}_fits.npz",allow_pickle=False) as original:
                truth=original["truth"].copy()
            saved={"truth":truth,"target":target,"media_slots":ms,"control_slots":cs,"saturation_anchor":data["saturation_anchor"][ms]}
            report={"world_id":world,"terms":names,"fits":{}}
            for method in ("bounded_estimated","bounded_true"):
                fixed=(corpus["carryover_alpha"][w,ms].astype(float),corpus["mm_kappa_mult"][w,ms].astype(float)) if method=="bounded_true" else None
                arrays,fit=fit_generator_bounded(x,z,data["saturation_anchor"][ms],target,box,fixed=fixed)
                components=np.zeros_like(truth);components[:,positions]=arrays["components"]
                coefficients=np.zeros(len(names));coefficients[positions]=arrays["coefficients"]
                arrays.update(components=components,coefficients=coefficients)
                np.testing.assert_allclose(components.sum(axis=1),arrays["prediction"],rtol=1e-12,atol=1e-12)
                fit["gates"]=score_bounded_components(components,truth,names,fit["success"])
                report["fits"][method]=fit
                saved.update({f"fit_{method}_{key}":value for key,value in arrays.items()})
            np.savez_compressed(array_path,**saved)
            write_study_json(report_path,report)
            reports.append(report)
            write_study_json(directory/"progress.json",{"completed":len(reports),"total":len(saved_study_records)})
        write_study_json(directory/"reports.json",reports)
        write_study_json(directory/"progress.json",{"completed":len(reports),"total":len(saved_study_records)})
        return reports


    return (run_generator_bounded_study,)


@app.cell(hide_code=True)
def generator_bounded_ols_results(run_generator_bounded_study):
    bounded_ols_reports=run_generator_bounded_study()
    return (bounded_ols_reports,)


@app.cell(hide_code=True)
def generator_bounded_ols_comparison(
    StudyPath,
    bounded_ols_reports,
    mo,
    np,
    saved_study_records,
    score_bounded_components,
    write_study_json,
):
    bounded_ols_comparison=[]
    for _relative in (.05,.1):
        for _absolute in (.005,.01):
            for _label,_method in (("Ordinary: estimated",0),("Generator-bounded: estimated","bounded_estimated"),("Ordinary: true transforms",2),("Generator-bounded: true transforms","bounded_true")):
                _passes=0;_converged=0
                for _record in saved_study_records:
                    _world=_record["world_id"]
                    if isinstance(_method,str):
                        _fit=bounded_ols_reports[_world]["fits"][_method]
                        _gate=next(r for r in _fit["gates"] if r["relative"]==_relative and r["absolute"]==_absolute)
                    else:
                        _fit=_record["fits"][_method]
                        with np.load(StudyPath("notebooks/ols_study_50_seed10029")/f"world_{_world:02d}_fits.npz",allow_pickle=False) as _f:
                            _gate=next(r for r in score_bounded_components(_f[f"fit_{_method}_components"],_f["truth"],_record["terms"],_fit["success"]) if r["relative"]==_relative and r["absolute"]==_absolute)
                    _converged+=int(_fit["success"]);_passes+=int(_gate["world_pass"])
                bounded_ols_comparison.append({"Method":_label,"Relative tolerance":_relative,"Outcome tolerance":_absolute,"Converged":_converged,"Recovery passes (of 50)":_passes})
    write_study_json(StudyPath("notebooks/ols_study_50_seed10029_generator_bounds/comparison.json"),bounded_ols_comparison)
    mo.vstack([mo.md("### Generator-support-bounded least squares\nNoisy outcomes only; all 50 existing worlds. Uses the known graph and the same saved reference anchors as MCMC. Estimated-transform fits jointly optimize baseline, reference amplitudes, controls, adstock and log saturation multiplier within generator boxes. True-transform fits use bounded linear least squares in reference-amplitude coordinates. **No generator noise likelihood or prior-density penalty is added.** Noiseless fits remain archived for validation, not in the scorecard selector. Nonconverged fits are retained and fail recovery regardless of point error. The nonlinear budget remains two starts ×120 evaluations; each evaluation is more expensive than an ordinary profiled step."),mo.ui.table(bounded_ols_comparison,selection=None)])

    return


@app.cell(hide_code=True)
def generator_bounded_ols_validation(
    StudyPath,
    bayesian_world_data,
    bounded_ols_reports,
    constant_baseline_cfg,
    generator_box_spec,
    mo,
    np,
    saved_study_records,
    score_bounded_components,
    solve_reference_box,
    write_study_json,
):
    bounded_ols_validation={"fits_checked":0,"max_numpy_component_difference":0.0,"fixed_linear_objective_checks":0}
    _box=generator_box_spec(constant_baseline_cfg)
    for _record in saved_study_records:
        _world=_record["world_id"]
        _data,_corpus=bayesian_world_data(_record);_w=_record["local_world"]
        with np.load(StudyPath("notebooks/ols_study_50_seed10029_generator_bounds")/f"world_{_world:02d}_fits.npz",allow_pickle=False) as _a:
            _ms=_a["media_slots"];_cs=_a["control_slots"];_m=len(_ms);_n=len(_cs)
            _positions=[0]+[_record["terms"].index(f"C{i+1}") for i in _ms]+[_record["terms"].index(f"Z{i+1}") for i in _cs]
            _null=np.setdiff1d(np.arange(len(_record["terms"])),_positions)
            for _method in ("bounded_estimated","bounded_true"):
                _prefix=f"fit_{_method}_"
                _weights=_a[_prefix+"reference_weights"];_alpha=_a[_prefix+"alpha"];_kappa=_a[_prefix+"kappa"]
                _lo=np.r_[_box["intercept"][0],np.full(_m,_box["amplitude"][0]),np.full(_n,_box["control_reference"][0])]
                _hi=np.r_[_box["intercept"][1],np.full(_m,_box["amplitude"][1]),np.full(_n,_box["control_reference"][1])]
                assert np.all(_weights>=_lo-1e-9) and np.all(_weights<=_hi+1e-9)
                assert np.all((_alpha>=_box["alpha"][0]-1e-7)&(_alpha<=_box["alpha"][1]+1e-7))
                assert np.all((_kappa>=_box["kappa"][0]-1e-7)&(_kappa<=_box["kappa"][1]+1e-7))
                _raw=_corpus["treatment_raw"][_w][:,_ms].astype(float)
                _carried=np.column_stack([np.convolve(_raw[:,j],_alpha[j]**np.arange(8)/np.sum(_alpha[j]**np.arange(8)))[:104] for j in range(_m)])[7:]
                _half=_kappa*_data["saturation_anchor"][_ms]
                _response=_carried/(_carried+_half)
                _design=np.column_stack([np.ones(97),_response*(1+_kappa),_corpus["covariates"][_w,7:][:,_cs]/_box["control_reference_scale"]])
                _expected=_design*_weights
                _actual=_a[_prefix+"components"][:,_positions]
                np.testing.assert_allclose(_actual,_expected,rtol=1e-10,atol=1e-10)
                assert np.all(_a[_prefix+"components"][:,_null]==0)
                np.testing.assert_allclose(_a[_prefix+"prediction"],_expected.sum(axis=1),rtol=1e-10,atol=1e-10)
                _fit=bounded_ols_reports[_world]["fits"][_method]
                assert _fit["gates"]==score_bounded_components(_a[_prefix+"components"],_a["truth"],_record["terms"],_fit["success"])
                bounded_ols_validation["fits_checked"]+=1
                bounded_ols_validation["max_numpy_component_difference"]=max(bounded_ols_validation["max_numpy_component_difference"],float(np.max(np.abs(_actual-_expected))))
                if _method=="bounded_true":
                    np.testing.assert_array_equal(_alpha,_corpus["carryover_alpha"][_w,_ms])
                    np.testing.assert_array_equal(_kappa,_corpus["mm_kappa_mult"][_w,_ms])
                    _free=np.linalg.lstsq(_design,_a["target"],rcond=None)[0]
                    _bounded_sse=np.sum((_design@_weights-_a["target"])**2)
                    _free_sse=np.sum((_design@_free-_a["target"])**2)
                    assert _bounded_sse>=_free_sse-1e-8
                    # A noiseless synthetic target validates the constrained linear solve only;
                    # it is not another dataset fit or scorecard comparator.
                    _interior=(_lo+_hi)/2
                    _recovered,_result=solve_reference_box(_design,_design@_interior,_lo,_hi)
                    assert _result.success
                    np.testing.assert_allclose(_design@_recovered,_design@_interior,rtol=1e-9,atol=1e-9)
                    bounded_ols_validation["fixed_linear_objective_checks"]+=1
    write_study_json(StudyPath("notebooks/ols_study_50_seed10029_generator_bounds/validation.json"),bounded_ols_validation)
    mo.md(f"Bounded OLS validation: **{bounded_ols_validation['fits_checked']} fits**, parameter supports and exact nulls checked; independent NumPy reconstruction agrees to {bounded_ols_validation['max_numpy_component_difference']:.2g}; all 50 bounded-linear checks passed.")

    return


@app.cell(hide_code=True)
def correlated_pair_recovery_functions(
    StudyPath,
    bayes_xr,
    bayesian_reports,
    bayesian_world_data,
    bounded_ols_reports,
    np,
    posterior_component_draws,
    saved_study_records,
    scorecard_geometry_by_world,
    write_study_json,
):
    def evaluate_correlated_media_pairs():
        directory=StudyPath("notebooks/correlated_media_pair_recovery")
        directory.mkdir(exist_ok=True)
        rows=[];point_rows=[];excluded=[]
        for record in saved_study_records:
            w=record["world_id"];names=record["terms"]
            report=bayesian_reports[w];ols_report=bounded_ols_reports[w]["fits"]["bounded_estimated"]
            with np.load(StudyPath("notebooks/bayesian_mmm_draw_diag")/f"world_{w:02d}_recovery.npz",allow_pickle=False) as f:
                truth=f["truth"].copy();nuts_mean=f["mean"].copy()
            with np.load(StudyPath("notebooks/ols_study_50_seed10029_generator_bounds")/f"world_{w:02d}_fits.npz",allow_pickle=False) as f:
                ols_mean=f["fit_bounded_estimated_components"].copy()
            ng=next(g for g in report["gates"] if g["relative"]==.1 and g["absolute"]==.01)
            og=next(g for g in ols_report["gates"] if g["relative"]==.1 and g["absolute"]==.01)
            media_indices=[i for i,n in enumerate(names) if n.startswith("C")]
            control_indices=[i for i,n in enumerate(names) if n.startswith("Z")]
            point={"World":w+1,"NUTS valid":report["diagnostics_pass"],"OLS valid":ols_report["success"],"NUTS point pass":ng["error_gates_pass"],"OLS point pass":og["error_gates_pass"],
                "NUTS failing media":sum(n.startswith("C") for n in ng["failing_channels"]),"NUTS failing controls":sum(n.startswith("Z") for n in ng["failing_channels"]),
                "OLS failing media":sum(n.startswith("C") for n in og["failing_channels"]),"OLS failing controls":sum(n.startswith("Z") for n in og["failing_channels"])}
            for label,indices in (("media",media_indices),("controls",control_indices),("all contributions",list(range(1,len(names))))):
                norm=np.linalg.norm(truth[:,indices])
                for method,mean in (("NUTS",nuts_mean),("OLS",ols_mean)):
                    point[f"{method} {label} relative error"]=float(np.linalg.norm(mean[:,indices]-truth[:,indices])/norm) if norm>0 else None
            point_rows.append(point)
            geometry=scorecard_geometry_by_world[w]
            media=[i for i,r in enumerate(geometry["rows"]) if r["Kind"]=="Media"]
            candidates=[(float(geometry["correlation"][i,j]),i,j) for i in media for j in media if i<j and np.isfinite(geometry["correlation"][i,j]) and geometry["correlation"][i,j]>0]
            if not candidates:
                excluded.append({"World":w+1,"Reason":"Fewer than two varying nonzero media with positive correlation"});continue
            correlation,i,j=max(candidates,key=lambda t:t[0])
            pair=[geometry["rows"][i]["Channel"],geometry["rows"][j]["Channel"]]
            indices=[names.index(n) for n in pair]
            truth_sum=truth[:,indices].sum(axis=1)
            data,_=bayesian_world_data(record)
            with bayes_xr.open_datatree(StudyPath("notebooks/bayesian_mmm_draw_diag")/f"world_{w:02d}_posterior.nc",engine="h5netcdf") as tree:
                components,_=posterior_component_draws(tree,data,record)
            pair_draws=components[:,:,indices].copy();del components
            sum_draws=pair_draws.sum(axis=2)
            sum_mean=sum_draws.mean(axis=0)
            np.testing.assert_allclose(sum_mean,nuts_mean[:,indices].sum(axis=1),rtol=1e-10,atol=1e-10)
            lower,upper=np.quantile(sum_draws,[.025,.975],axis=0)
            marginal_lower,marginal_upper=np.quantile(pair_draws,[.025,.975],axis=0)
            radius=float(np.quantile(np.linalg.norm(sum_draws-sum_mean,axis=1),.95))
            marginal_width_sum=(marginal_upper-marginal_lower).sum(axis=1)
            positive_width=marginal_width_sum>0
            width_ratio=float(np.median((upper-lower)[positive_width]/marginal_width_sum[positive_width])) if positive_width.any() else None
            means=pair_draws.mean(axis=1)
            posterior_correlation=float(np.corrcoef(means.T)[0,1])
            np.savez_compressed(directory/f"world_{w:02d}_pair.npz",indices=np.array(indices),truth=truth_sum,nuts_mean=sum_mean,nuts_lower95=lower,nuts_upper95=upper,ols_mean=ols_mean[:,indices].sum(axis=1))
            for relative in (.05,.1):
                for absolute in (.005,.01):
                    allowance=max(relative*np.linalg.norm(truth_sum),absolute*np.linalg.norm(truth.sum(axis=1)))
                    for method,mean,valid,gates in (("NUTS",nuts_mean,report["diagnostics_pass"],report["gates"]),("Bounded OLS",ols_mean,ols_report["success"],ols_report["gates"])):
                        gate=next(g for g in gates if g["relative"]==relative and g["absolute"]==absolute)
                        fail=[n for n in pair if n in gate["failing_channels"]]
                        outside_fail=[n for n in gate["failing_channels"] if n not in pair]
                        errors=mean[:,indices]-truth[:,indices]
                        error=float(np.linalg.norm(errors.sum(axis=1)))
                        sum_pass=bool(error<=allowance)
                        denom=np.linalg.norm(errors[:,0])+np.linalg.norm(errors[:,1])
                        rows.append({"World":w+1,"Method":method,"Pair":" + ".join(pair),"True correlation":correlation,"Relative tolerance":relative,"Outcome tolerance":absolute,"Computationally valid":bool(valid),
                            "Pair members failing":len(fail),"Failing members":", ".join(fail),"Other failing channels":", ".join(outside_fail),"Sum point pass":sum_pass,"Sum error / tolerance":error/allowance,
                            "Error cancellation ratio":float(error/denom) if denom>0 else None,
                            "World point pass before":gate["error_gates_pass"],"World pass after grouping":bool(valid and sum_pass and not outside_fail),
                            "Sum 95% radius / tolerance":radius/allowance if method=="NUTS" else None,
                            "Sum interval width / sum of marginal widths":width_ratio if method=="NUTS" else None,
                            "Posterior correlation of time-mean contributions":posterior_correlation if method=="NUTS" else None})
        write_study_json(directory/"results.json",{"selection":"Largest positive Pearson correlation among true nonzero, nonconstant media; independent of recovery errors. No minimum correlation cutoff. Constants excluded as undefined.","point_comparison":point_rows,"pairs":rows,"excluded_worlds":excluded})
        return point_rows,rows,excluded


    return (evaluate_correlated_media_pairs,)


@app.cell(hide_code=True)
def correlated_pair_recovery_results(evaluate_correlated_media_pairs):
    paired_mean_recovery,correlated_pair_recovery,pair_excluded_worlds=evaluate_correlated_media_pairs()
    return correlated_pair_recovery, paired_mean_recovery


@app.cell(hide_code=True)
def correlated_pair_recovery_summary(
    StudyPath,
    correlated_pair_recovery,
    mo,
    np,
    paired_mean_recovery,
    write_study_json,
):
    _common=[r for r in paired_mean_recovery if r["NUTS valid"] and r["OLS valid"]]
    paired_mean_summary={"jointly_valid_worlds":len(_common),"NUTS_passes":sum(r["NUTS point pass"] for r in _common),"bounded_OLS_passes":sum(r["OLS point pass"] for r in _common),
        "NUTS_lower_media_error":sum(r["NUTS media relative error"]<r["OLS media relative error"] for r in _common),
        "median_paired_media_error_ratio":float(np.median([r["NUTS media relative error"]/r["OLS media relative error"] for r in _common]))}
    pair_failure_counts=[]
    for _method in ("NUTS","OLS"):
        _failed=[r for r in paired_mean_recovery if r[_method+" valid"] and not r[_method+" point pass"]]
        pair_failure_counts.append({"Method":_method,"Valid fits failing recovery":len(_failed),"Exactly one media fails":sum(r[_method+" failing media"]==1 for r in _failed),
            "Exactly two media fail":sum(r[_method+" failing media"]==2 for r in _failed),"Three or more media fail":sum(r[_method+" failing media"]>=3 for r in _failed),
            "Only one media fails; no controls fail":sum(r[_method+" failing media"]==1 and r[_method+" failing controls"]==0 for r in _failed)})
    pair_primary_rows=[r for r in correlated_pair_recovery if r["Relative tolerance"]==.1 and r["Outcome tolerance"]==.01]
    pair_recovery_summary=[]
    for _method in ("NUTS","Bounded OLS"):
        _eligible=[r for r in pair_primary_rows if r["Method"]==_method and r["Computationally valid"]]
        _broken=[r for r in _eligible if r["Pair members failing"]>0]
        _rescued=[r for r in _broken if r["Sum point pass"]]
        pair_recovery_summary.append({"Method":_method,"Valid worlds with eligible pair":len(_eligible),"Pairs with ≥1 member failing":len(_broken),
            "Of those, sum passes":len(_rescued),"Pairs with both failing":sum(r["Pair members failing"]==2 for r in _broken),
            "Both fail but sum passes":sum(r["Pair members failing"]==2 for r in _rescued),
            "Previously failing worlds rescued by grouping":sum(r["World pass after grouping"] and not r["World point pass before"] for r in _eligible)})
    write_study_json(StudyPath("notebooks/correlated_media_pair_recovery/summary.json"),{"point_comparison":paired_mean_summary,"failure_counts":pair_failure_counts,"pair_recovery":pair_recovery_summary})
    mo.vstack([mo.md(f"### Posterior mean versus bounded estimated-transform OLS\nAt 10% / 1%, among **{len(_common)} worlds where both computations are valid**, NUTS passes {paired_mean_summary['NUTS_passes']} and bounded OLS passes {paired_mean_summary['bounded_OLS_passes']}. NUTS has lower media-group trajectory error in {paired_mean_summary['NUTS_lower_media_error']} worlds; the median within-world error ratio is {paired_mean_summary['median_paired_media_error_ratio']:.3f}. This is point recovery, not posterior precision."),
        mo.ui.table(pair_failure_counts,selection=None),mo.md("### Does summing the most correlated media pair help?\nChoose the largest **positive Pearson correlation of true contributions**, independently of fit errors, among nonzero media. No minimum correlation is imposed, so the best pair may still be weakly correlated. Fewer than two varying media or no positive pair is ineligible. The sum uses max(10% × true-sum L2 norm, 1% × true noiseless-outcome L2 norm). This can pass because errors cancel **or because the sum has a larger allowance**; it is not by itself proof of identifiability. Each method's summary excludes its computational failures. Original individual scores are unchanged."),
        mo.ui.table(pair_recovery_summary,selection=None),mo.ui.table(pair_primary_rows,selection=None)])

    return (pair_primary_rows,)


@app.cell(hide_code=True)
def total_media_recovery_functions(
    StudyPath,
    bayes_xr,
    bayesian_reports,
    bayesian_world_data,
    bounded_ols_reports,
    np,
    posterior_component_draws,
    saved_study_records,
    scorecard_geometry_by_world,
    write_study_json,
):
    def inspect_media_blocks():
        blocks=[];conditioning=[]
        for record in saved_study_records:
            w=record["world_id"];geometry=scorecard_geometry_by_world[w]
            ix=[i for i,r in enumerate(geometry["rows"]) if r["Kind"]=="Media"]
            names=[geometry["rows"][i]["Channel"] for i in ix]
            corr=geometry["correlation"][np.ix_(ix,ix)]
            with np.load(StudyPath("notebooks/ols_study_50_seed10029")/f"world_{w:02d}_fits.npz",allow_pickle=False) as f:
                x=f["truth"][:,[record["terms"].index(n) for n in names]]
            centered=x-x.mean(axis=0)
            norms=np.linalg.norm(centered,axis=0);keep=norms>0
            standardized=centered[:,keep]/norms[keep]
            s=np.linalg.svd(standardized,compute_uv=False)
            condition=float(s[0]/s[-1]) if len(s) and s[-1]>0 else None
            conditioning.append({"World":w+1,"Varying media":int(keep.sum()),"Centered standardized design condition number":condition})
            for absolute in (False,True):
                for threshold in (.7,.8,.9):
                    edges=(np.abs(corr) if absolute else corr)>=threshold
                    np.fill_diagonal(edges,False)
                    unseen=set(range(len(ix)))
                    while unseen:
                        stack=[unseen.pop()];component=[]
                        while stack:
                            i=stack.pop();component.append(i)
                            neighbors=set(np.flatnonzero(edges[i]))&unseen
                            unseen-=neighbors;stack.extend(neighbors)
                        if len(component)<3:continue
                        component=sorted(component)
                        sub=corr[np.ix_(component,component)];values=sub[np.triu_indices(len(component),1)]
                        blocks.append({"World":w+1,"Edge rule":"absolute correlation" if absolute else "positive correlation","Threshold":threshold,"Channels":", ".join(names[i] for i in component),"Size":len(component),
                            "Minimum within-block absolute correlation":float(np.min(np.abs(values))),"Minimum signed correlation":float(np.min(values)),"Maximum signed correlation":float(np.max(values)),
                            "All pairs meet threshold":bool(np.all((np.abs(values) if absolute else values)>=threshold))})
        return blocks,conditioning


    def evaluate_total_media():
        directory=StudyPath("notebooks/total_media_recovery");directory.mkdir(exist_ok=True)
        rows=[]
        for record in saved_study_records:
            w=record["world_id"];names=record["terms"]
            report=bayesian_reports[w];ols_report=bounded_ols_reports[w]["fits"]["bounded_estimated"]
            indices=[i for i,n in enumerate(names) if n.startswith("C")]
            with np.load(StudyPath("notebooks/bayesian_mmm_draw_diag")/f"world_{w:02d}_recovery.npz",allow_pickle=False) as f:
                truth=f["truth"].copy();nuts_mean=f["mean"].copy()
            with np.load(StudyPath("notebooks/ols_study_50_seed10029_generator_bounds")/f"world_{w:02d}_fits.npz",allow_pickle=False) as f:
                ols_mean=f["fit_bounded_estimated_components"].copy()
            data,_=bayesian_world_data(record)
            with bayes_xr.open_datatree(StudyPath("notebooks/bayesian_mmm_draw_diag")/f"world_{w:02d}_posterior.nc",engine="h5netcdf") as tree:
                components,_=posterior_component_draws(tree,data,record)
            media_draws=components[:,:,indices]
            total_draws=media_draws.sum(axis=2)
            total_mean=total_draws.mean(axis=0);total_truth=truth[:,indices].sum(axis=1)
            lower,upper=np.quantile(total_draws,[.025,.975],axis=0)
            radius=float(np.quantile(np.linalg.norm(total_draws-total_mean,axis=1),.95))
            member_radii=np.quantile(np.linalg.norm(media_draws-nuts_mean[:,indices],axis=1),.95,axis=0)
            marginal_lower,marginal_upper=np.quantile(media_draws,[.025,.975],axis=0)
            marginal_width_sum=(marginal_upper-marginal_lower).sum(axis=1)
            positive_width=marginal_width_sum>0
            width_ratio=float(np.median((upper-lower)[positive_width]/marginal_width_sum[positive_width])) if positive_width.any() else None
            intercept_correlation=float(np.corrcoef(total_draws.mean(axis=1),components[:,0,0])[0,1])
            del components,media_draws
            np.testing.assert_allclose(total_mean,nuts_mean[:,indices].sum(axis=1),rtol=1e-10,atol=1e-10)
            np.savez_compressed(directory/f"world_{w:02d}_total_media.npz",truth=total_truth,nuts_mean=total_mean,nuts_lower95=lower,nuts_upper95=upper,ols_mean=ols_mean[:,indices].sum(axis=1))
            for relative in (.05,.1):
                for absolute in (.005,.01):
                    allowance=max(relative*np.linalg.norm(total_truth),absolute*np.linalg.norm(truth.sum(axis=1)))
                    member_allowances=np.maximum(relative*np.linalg.norm(truth[:,indices],axis=0),absolute*np.linalg.norm(truth.sum(axis=1)))
                    for method,mean,valid,gates in (("NUTS",nuts_mean,report["diagnostics_pass"],report["gates"]),("Bounded OLS",ols_mean,ols_report["success"],ols_report["gates"])):
                        gate=next(g for g in gates if g["relative"]==relative and g["absolute"]==absolute)
                        errors=mean[:,indices]-truth[:,indices];total_error=float(np.linalg.norm(errors.sum(axis=1)))
                        denom=float(np.linalg.norm(errors,axis=0).sum())
                        rows.append({"World":w+1,"Method":method,"Relative tolerance":relative,"Outcome tolerance":absolute,"Computationally valid":bool(valid),
                            "All individual media pass":not any(n.startswith("C") for n in gate["failing_channels"]),"Total media point pass":bool(total_error<=allowance),
                            "Total media relative error":total_error/np.linalg.norm(total_truth),"Total media error / tolerance":total_error/allowance,
                            "Error cancellation ratio":total_error/denom if denom>0 else None,
                            "Total media 95% radius / tolerance":radius/allowance if method=="NUTS" else None,
                            "Largest individual media 95% radius / tolerance":float(np.max(member_radii/member_allowances)) if method=="NUTS" else None,
                            "Total interval width / sum of marginal widths":width_ratio if method=="NUTS" else None,
                            "Posterior correlation: media level vs intercept":intercept_correlation if method=="NUTS" else None})
        write_study_json(directory/"results.json",rows)
        return rows


    return evaluate_total_media, inspect_media_blocks


@app.cell(hide_code=True)
def total_media_recovery_results(evaluate_total_media, inspect_media_blocks):
    media_correlation_blocks,media_design_conditioning=inspect_media_blocks()
    total_media_recovery=evaluate_total_media()
    return (
        media_correlation_blocks,
        media_design_conditioning,
        total_media_recovery,
    )


@app.cell(hide_code=True)
def total_media_recovery_summary(
    StudyPath,
    media_correlation_blocks,
    media_design_conditioning,
    mo,
    np,
    paired_mean_recovery,
    total_media_recovery,
    write_study_json,
):
    total_media_primary=[r for r in total_media_recovery if r["Relative tolerance"]==.1 and r["Outcome tolerance"]==.01]
    _common={r["World"] for r in paired_mean_recovery if r["NUTS valid"] and r["OLS valid"]}
    total_media_summary=[]
    for _method in ("NUTS","Bounded OLS"):
        _rows=[r for r in total_media_primary if r["World"] in _common and r["Method"]==_method]
        total_media_summary.append({"Method":_method,"Jointly valid worlds":len(_rows),"All individual media pass":sum(r["All individual media pass"] for r in _rows),
            "Total media point passes":sum(r["Total media point pass"] for r in _rows),"Median total-media relative error":float(np.median([r["Total media relative error"] for r in _rows]))})
    _nuts=[r for r in total_media_primary if r["World"] in _common and r["Method"]=="NUTS"]
    total_media_precision={"NUTS worlds":len(_nuts),"Total radius within tolerance":sum(r["Total media 95% radius / tolerance"]<=1 for r in _nuts),
        "All individual radii within tolerance":sum(r["Largest individual media 95% radius / tolerance"]<=1 for r in _nuts),
        "Median total radius / tolerance":float(np.median([r["Total media 95% radius / tolerance"] for r in _nuts])),
        "Median largest individual radius / tolerance":float(np.median([r["Largest individual media 95% radius / tolerance"] for r in _nuts])),
        "Median media-level/intercept posterior correlation":float(np.median([r["Posterior correlation: media level vs intercept"] for r in _nuts]))}
    media_block_frequency=[]
    for _rule in ("positive correlation","absolute correlation"):
        for _threshold in (.7,.8,.9):
            _blocks=[r for r in media_correlation_blocks if r["Edge rule"]==_rule and r["Threshold"]==_threshold]
            media_block_frequency.append({"Edge rule":_rule,"Threshold":_threshold,"Worlds with ≥3-channel connected block":len({r["World"] for r in _blocks}),"Largest block":max((r["Size"] for r in _blocks),default=0)})
    write_study_json(StudyPath("notebooks/total_media_recovery/summary.json"),{"point_recovery":total_media_summary,"posterior_precision":total_media_precision,"block_frequency":media_block_frequency,"blocks":media_correlation_blocks,"conditioning":media_design_conditioning})
    mo.vstack([mo.md("### Total media versus individual media recovery\nUses **generator-bounded estimated-transform OLS** and NUTS posterior means, on the same 44 computationally valid worlds. Gate: 10% of the true total-media trajectory norm or 1% of the noiseless outcome norm, whichever is larger. Grouping changes the estimand and its allowance; it does not retroactively fix individual attribution."),
        mo.ui.table(total_media_summary,selection=None),mo.ui.table([total_media_precision],selection=None),mo.md("The 95% radius measures joint-draw trajectory variation around the posterior mean. A smaller radius/tolerance indicates more precise recovery of the requested estimand; it is not the same as point accuracy or pointwise interval coverage. OLS has **no uncertainty estimate** in this analysis. Media-level/intercept correlation describes a posterior trade-off, not by itself proof of non-identifiability."),
        mo.md("### Are there larger true-media correlation blocks?\nEdges connect channels whose true Pearson correlation (or absolute correlation) reaches the displayed threshold. Connected blocks may be transitive chains: **not every pair necessarily meets the threshold**. Negative edges are included only in the absolute-correlation analysis. Centered standardized design conditioning also checks joint dependence, but excludes intercept confounding and does not characterize the full nonlinear model."),
        mo.ui.table(media_block_frequency,selection=None),mo.ui.table(media_correlation_blocks,selection=None),mo.ui.table(media_design_conditioning,selection=None)])

    return (total_media_primary,)


@app.cell(hide_code=True)
def mcmc_difficulty_profile(
    StudyPath,
    bayes_xr,
    bayesian_reports,
    media_design_conditioning,
    np,
    pg,
    saved_study_records,
    scorecard_geometry_by_world,
    total_media_primary,
    write_study_json,
):
    def profile_recovery_difficulty():
        worlds=[];channels=[]
        for record in saved_study_records:
            w=record["world_id"];names=record["terms"];report=bayesian_reports[w]
            geometry=scorecard_geometry_by_world[w]
            media_rows=[r for r in geometry["rows"] if r["Kind"]=="Media"]
            slots=[int(r["Channel"][1:])-1 for r in media_rows]
            indices=[names.index(r["Channel"]) for r in media_rows]
            with bayes_xr.open_datatree(StudyPath("notebooks/bayesian_mmm_draw_diag")/f"world_{w:02d}_posterior.nc",engine="h5netcdf") as tree:
                intercept=tree["posterior"]["intercept"].values.reshape(-1)
                levels=tree["posterior"]["mean_media_contribution"].transpose("chain","draw","media").values.reshape(-1,15)[:,slots]
            correlations=np.array([np.corrcoef(intercept,levels[:,j])[0,1] for j in range(len(slots))])
            with np.load(StudyPath("notebooks/bayesian_mmm_draw_diag")/f"world_{w:02d}_recovery.npz",allow_pickle=False) as a:
                truth=a["truth"].copy();mean=a["mean"].copy()
            corpus=pg.load_corpus(StudyPath("notebooks/ols_study_50_seed10029")/f"batch_{record['batch']:02d}.npz")
            noise_sd=float(np.std(corpus["outcome_noise"][record["local_world"],7:]))
            allowance=np.maximum(.1*np.linalg.norm(truth[:,indices],axis=0),.01*np.linalg.norm(truth.sum(axis=1)))
            error_ratios=np.linalg.norm(mean[:,indices]-truth[:,indices],axis=0)/allowance
            gate=next(g for g in report["gates"] if g["relative"]==.1 and g["absolute"]==.01)
            for j,row in enumerate(media_rows):
                channels.append({"World":w+1,"Channel":row["Channel"],"MCMC valid":report["diagnostics_pass"],"Intercept posterior correlation":float(correlations[j]),
                    "True CV":row["CV"],"True temporal SD / noise SD":row["SD"]/noise_sd,"Point error / tolerance":float(error_ratios[j]),
                    "Posterior SD of time-mean contribution":float(np.std(levels[:,j]))})
            geo_media=[i for i,r in enumerate(geometry["rows"]) if r["Kind"]=="Media"]
            sub=geometry["correlation"][np.ix_(geo_media,geo_media)];corr_values=sub[np.triu_indices(len(geo_media),1)]
            total=next(r for r in total_media_primary if r["World"]==w+1 and r["Method"]=="NUTS")
            cv=[r["CV"] for r in media_rows if r["CV"] is not None]
            total_truth=truth[:,indices].sum(axis=1)
            worlds.append({"World":w+1,"MCMC valid":report["diagnostics_pass"],"Media effects":len(slots),"Control effects":record["nonzero_control_effects"],
                "Minimum media CV":min(cv) if cv else None,"Median media CV":float(np.median(cv)) if cv else None,"Near-flat media (CV ≤10%)":sum(c<=.1 for c in cv),
                "Total media CV":float(np.std(total_truth)/abs(np.mean(total_truth))),"Minimum media temporal SD / noise SD":min(r["SD"] for r in media_rows)/noise_sd,
                "Median media temporal SD / noise SD":float(np.median([r["SD"] for r in media_rows]))/noise_sd,"Realized noise SD":noise_sd,
                "Maximum absolute media correlation":float(np.nanmax(np.abs(corr_values))) if len(corr_values) else None,
                "Standardized media design condition number":media_design_conditioning[w]["Centered standardized design condition number"],
                "Most negative media/intercept posterior correlation":float(correlations.min()),"Strongly negative channels (r ≤−0.8)":int(np.sum(correlations<=-.8)),
                "Posterior intercept SD":float(np.std(intercept)),"Worst media error / tolerance":float(error_ratios.max()),
                "All individual media pass":bool(np.all(error_ratios<=1)),"Failing media":", ".join(n for n in gate["failing_channels"] if n.startswith("C")),
                "Total media point pass":total["Total media point pass"],"Total media error / tolerance":total["Total media error / tolerance"],
                "Worst media 95% radius / tolerance":total["Largest individual media 95% radius / tolerance"],"Total media 95% radius / tolerance":total["Total media 95% radius / tolerance"]})
        path=StudyPath("notebooks/mcmc_difficulty_profile");path.mkdir(exist_ok=True)
        write_study_json(path/"results.json",{"worlds":worlds,"channels":channels,"definition":"Channel/intercept correlations are across joint posterior draws of time-average channel contributions, NOT time-series correlations with the constant baseline. Recovery uses 10%/1% tolerance. Computational failures retained but excluded from associations and main frequencies."})
        return worlds,channels

    mcmc_world_difficulty,mcmc_channel_difficulty=profile_recovery_difficulty()

    return mcmc_channel_difficulty, mcmc_world_difficulty


@app.cell(hide_code=True)
def mcmc_difficulty_summary(
    StudyPath,
    go,
    make_subplots,
    mcmc_channel_difficulty,
    mcmc_world_difficulty,
    mo,
    write_study_json,
):
    from scipy.stats import spearmanr as difficulty_spearmanr

    _valid=[r for r in mcmc_world_difficulty if r["MCMC valid"]]
    _channels=[r for r in mcmc_channel_difficulty if r["MCMC valid"]]
    mcmc_baseline_frequency=[]
    for _threshold in (-.7,-.8,-.9):
        mcmc_baseline_frequency.append({"Posterior correlation ≤":_threshold,"Worlds with ≥1 channel":sum(r["Most negative media/intercept posterior correlation"]<=_threshold for r in _valid),"Valid worlds":len(_valid),
            "Channels":sum(r["Intercept posterior correlation"]<=_threshold for r in _channels),"Valid nonzero media channels":len(_channels)})
    mcmc_difficulty_strata=[]
    for _label,_subset in (("1–2 media",[r for r in _valid if r["Media effects"]<=2]),("3–5 media",[r for r in _valid if 3<=r["Media effects"]<=5]),("6+ media",[r for r in _valid if r["Media effects"]>=6]),
        ("At least one near-flat media (CV ≤10%)",[r for r in _valid if r["Near-flat media (CV ≤10%)"]>0]),("No near-flat media",[r for r in _valid if r["Near-flat media (CV ≤10%)"]==0]),
        ("Any channel/intercept r ≤−0.8",[r for r in _valid if r["Strongly negative channels (r ≤−0.8)"]>0]),("No channel/intercept r ≤−0.8",[r for r in _valid if r["Strongly negative channels (r ≤−0.8)"]==0])):
        mcmc_difficulty_strata.append({"Subset":_label,"Worlds":len(_subset),"Individual-media recovery failures":sum(not r["All individual media pass"] for r in _subset),"Total-media recovery failures":sum(not r["Total media point pass"] for r in _subset)})
    mcmc_difficulty_associations=[]
    for _feature in ("Media effects","Control effects","Minimum media CV","Median media CV","Near-flat media (CV ≤10%)","Total media CV","Minimum media temporal SD / noise SD","Median media temporal SD / noise SD","Realized noise SD","Maximum absolute media correlation","Standardized media design condition number","Most negative media/intercept posterior correlation","Posterior intercept SD"):
        _rows=[r for r in _valid if r[_feature] is not None]
        _association={"Feature":_feature,"Worlds":len(_rows)}
        for _target in ("Worst media error / tolerance","Total media error / tolerance","Worst media 95% radius / tolerance","Total media 95% radius / tolerance"):
            _association[_target]=float(difficulty_spearmanr([r[_feature] for r in _rows],[r[_target] for r in _rows]).statistic)
        mcmc_difficulty_associations.append(_association)
    mcmc_hardness_rankings=[]
    for _metric in ("Worst media error / tolerance","Total media error / tolerance","Worst media 95% radius / tolerance","Total media 95% radius / tolerance"):
        for _rank,_row in enumerate(sorted(_valid,key=lambda r:r[_metric],reverse=True)[:5],start=1):
            mcmc_hardness_rankings.append({"Ranking metric":_metric,"Rank":_rank,"World":_row["World"],"Value":_row[_metric],"Media effects":_row["Media effects"]})
    write_study_json(StudyPath("notebooks/mcmc_difficulty_profile/summary.json"),{"baseline_frequency":mcmc_baseline_frequency,"strata":mcmc_difficulty_strata,"spearman_associations":mcmc_difficulty_associations,"rankings":mcmc_hardness_rankings})
    _difficulty_fig=make_subplots(rows=1,cols=2,horizontal_spacing=.14,subplot_titles=("Individual attribution: least variable channel","Total media: variability of the total"))
    for _col,_xkey,_ykey in ((1,"Minimum media CV","Worst media error / tolerance"),(2,"Total media CV","Total media error / tolerance")):
        _difficulty_fig.add_trace(go.Scatter(x=[r[_xkey] for r in _valid],y=[r[_ykey] for r in _valid],mode="markers",marker={"color":[r["Media effects"] for r in _valid],"coloraxis":"coloraxis","size":9},
            text=[f"World {r['World']} · {r['Media effects']} nonzero media" for r in _valid],hovertemplate="%{text}<br>CV %{x:.1%}<br>Error/tolerance %{y:.2f}<extra></extra>",showlegend=False),row=1,col=_col)
        _difficulty_fig.add_hline(y=1,line_dash="dash",row=1,col=_col)
        _difficulty_fig.update_xaxes(type="log",tickformat=".0%",title_text="True temporal CV",row=1,col=_col)
        _difficulty_fig.update_yaxes(type="log",title_text="Point error / tolerance",row=1,col=_col)
    _difficulty_fig.update_layout(height=460,template="plotly_white",coloraxis={"colorscale":"Viridis","colorbar":{"title":"Media count"}},title=f"Exploratory difficulty patterns · {len(_valid)} diagnostically valid worlds")
    mo.vstack([mo.md(f"### Which worlds are difficult for MCMC?\nAt 10% / 1% tolerance, **{sum(not r['All individual media pass'] for r in _valid)}/{len(_valid)}** diagnostically valid worlds fail individual-media recovery; **{sum(not r['Total media point pass'] for r in _valid)}/{len(_valid)}** fail total-media recovery. Sampling failures (worlds {', '.join(str(r['World']) for r in mcmc_world_difficulty if not r['MCMC valid']) or 'none'}) remain unresolved and are excluded here. Difficulty means attribution accuracy/precision under this data/model/prior setup, not merely slow or failed sampling. All fits already have oracle graph/reference information."),
        mo.ui.plotly(_difficulty_fig),mo.ui.table(mcmc_difficulty_strata,selection=None),mo.md(f"### Channel–baseline trade-offs\nThese correlations are **across joint posterior draws**, between the intercept and a channel's time-average contribution. They are not temporal correlations with the constant baseline. A strong correlation can exist in a narrow, accurate posterior: **{sum(r['All individual media pass'] for r in _valid if r['Strongly negative channels (r ≤−0.8)']>0)} of the {sum(r['Strongly negative channels (r ≤−0.8)']>0 for r in _valid)}** worlds with a channel correlation ≤−0.8 still pass every individual-media point gate. Marginal channel/intercept correlations can also be modest when uncertainty is spread across many channels. Use error and uncertainty alongside correlation."),
        mo.ui.table(mcmc_baseline_frequency,selection=None),mo.ui.table(mcmc_hardness_rankings,selection=None),mo.md(f"### Exploratory associations (Spearman rank correlation)\nThese are **unadjusted descriptive associations on {len(_valid)} worlds, not validated predictors or causal effects**. Channel count, minimum CV and signal-to-noise co-vary; more channels also create more opportunities to fail an all-channel gate. True-trajectory features are oracle diagnostics, whereas posterior correlations and intercept SD require an MCMC fit. Pairwise media correlation is undefined for a single channel, so that association uses a smaller subset."),mo.ui.table(mcmc_difficulty_associations,selection=None)])

    return


@app.cell(hide_code=True)
def scorecard_controls(mo):
    scorecard_flatness_cutoff=mo.ui.slider(start=0,stop=50,step=1,value=10,label="Near-flat CV cutoff (%)",show_value=True)
    scorecard_ols_choice=mo.ui.dropdown(options={"Generator-bounded · estimated transforms":"bounded_estimated","Generator-bounded · true transforms":"bounded_true","Ordinary OLS · estimated transforms":0,"Ordinary OLS · true transforms":2},value="Generator-bounded · estimated transforms",label="OLS comparison")
    scorecard_advi_choice=mo.ui.dropdown(options={"Off":None,"Full-rank ADVI seed 1":"fullrank_advi_1","Full-rank ADVI seed 2":"fullrank_advi_2"},value="Off",label="ADVI overlay (worlds 1–5 only)")
    mo.vstack([mo.md("## World scorecard\nUses the MCMC world selector above. Flatness is a descriptive warning, **not a recovery gate**. Constant nonzero contributions cannot be separated from the intercept by their time pattern; near-flat ones may be weakly distinguished. Neither warning removes data."),mo.hstack([scorecard_ols_choice,scorecard_advi_choice]),scorecard_flatness_cutoff])

    return (
        scorecard_advi_choice,
        scorecard_flatness_cutoff,
        scorecard_ols_choice,
    )


@app.cell(hide_code=True)
def scorecard_correlation_view(
    bayesian_world_picker,
    flatness_status,
    go,
    make_subplots,
    mo,
    scorecard_flatness_cutoff,
    scorecard_geometry_by_world,
):
    scorecard_geometry=scorecard_geometry_by_world[bayesian_world_picker.value]
    _cutoff=scorecard_flatness_cutoff.value/100
    _rows=scorecard_geometry["rows"]
    _labels=[r["Channel"] for r in _rows]
    _status=[flatness_status(r,_cutoff) for r in _rows]
    _cv_color=[0 if r["Constant"] else min(r["CV"],max(2*_cutoff,.01)) if r["CV"] is not None else None for r in _rows]
    _cv_text=[("constant" if r["Constant"] else f"{r['CV']:.1%}" if r["CV"] is not None else "undefined")+ (" ⚑" if s in ("Constant","Near-flat") else "") for r,s in zip(_rows,_status)]
    scorecard_correlation_figure=make_subplots(rows=1,cols=2,column_widths=[.82,.18],horizontal_spacing=.12,shared_yaxes=True,subplot_titles=("True contribution Pearson correlation","Temporal CV / flatness"))
    scorecard_correlation_figure.add_trace(go.Heatmap(z=scorecard_geometry["correlation"],x=_labels,y=_labels,zmin=-1,zmax=1,colorscale="RdBu",reversescale=True,
        colorbar={"orientation":"h","y":-.25,"len":.65,"x":.35},hoverongaps=False,hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>"),row=1,col=1)
    scorecard_correlation_figure.add_trace(go.Heatmap(z=[[v] for v in _cv_color],x=["CV"],y=_labels,text=[[v] for v in _cv_text],
        customdata=[[[r['Mean'],r['SD'],s]] for r,s in zip(_rows,_status)],colorscale="RdYlGn",zmin=0,zmax=max(2*_cutoff,.01),showscale=False,
        hovertemplate="%{y}: %{text}<br>Mean %{customdata[0]:.4g}<br>SD %{customdata[1]:.4g}<br>%{customdata[2]}<extra></extra>"),row=1,col=2)
    scorecard_correlation_figure.add_trace(go.Scatter(x=["CV"]*len(_labels),y=_labels,mode="text",text=_cv_text,textfont={"color":"black","size":11},showlegend=False,hoverinfo="skip"),row=1,col=2)
    scorecard_correlation_figure.update_yaxes(autorange="reversed")
    scorecard_correlation_figure.update_layout(height=max(430,32*len(_labels)+180),template="plotly_white",margin={"b":110},title=f"World {bayesian_world_picker.value+1}: contribution geometry (weeks 7–103)")
    scorecard_flatness_frequencies=[]
    for _kind in ("Media","Control","All"):
        _all=[(w,r) for w,g in scorecard_geometry_by_world.items() for r in g["rows"] if _kind=="All" or r["Kind"]==_kind]
        _constant=[(w,r) for w,r in _all if r["Constant"]]
        _near=[(w,r) for w,r in _all if flatness_status(r,_cutoff)=="Near-flat"]
        _flagged=_constant+_near
        scorecard_flatness_frequencies.append({"Contributions":_kind,"Count":len(_all),"Exactly constant":len(_constant),"Near-flat (not constant)":len(_near),
            "Flagged fraction":len(_flagged)/len(_all) if _all else None,"Worlds with a flag (of 50)":len({w for w,r in _flagged})})
    mo.vstack([mo.ui.plotly(scorecard_correlation_figure),mo.md("CV = temporal SD / |temporal mean|. **⚑** marks an exactly constant series or CV at/below the selected cutoff. A blank correlation is undefined, not zero. CV is undefined at zero mean and can be large for sign-changing controls. Color in the CV column clips at twice the cutoff; text shows the actual value. Known-null channels and the intercept are excluded. These are **oracle truth-based** descriptors—not diagnostics available directly from observed sales."),
        mo.ui.table([{**r,"Status":s} for r,s in zip(_rows,_status)],selection=None),mo.md("### Frequency across all 50 worlds"),mo.ui.table(scorecard_flatness_frequencies,selection=None)])

    return (scorecard_geometry,)


@app.cell(hide_code=True)
def whole_world_scorecard(
    StudyPath,
    bayes_xr,
    bayesian_reports,
    bayesian_world_picker,
    bounded_ols_reports,
    flatness_status,
    go,
    make_subplots,
    mo,
    np,
    pg,
    saved_study_records,
    scorecard_advi_choice,
    scorecard_flatness_cutoff,
    scorecard_geometry,
    scorecard_ols_choice,
):
    _world=bayesian_world_picker.value
    _record=saved_study_records[_world]
    _report=bayesian_reports[_world]
    with np.load(StudyPath("notebooks/bayesian_mmm_draw_diag")/f"world_{_world:02d}_recovery.npz",allow_pickle=False) as _f:
        _nuts={k:_f[k] for k in _f.files}
    with np.load(StudyPath("notebooks/ols_study_50_seed10029")/f"world_{_world:02d}_fits.npz",allow_pickle=False) as _f:
        _ols={k:_f[k] for k in _f.files}
    with bayes_xr.open_datatree(StudyPath("notebooks/bayesian_mmm_draw_diag")/f"world_{_world:02d}_posterior.nc",engine="h5netcdf") as _f:
        _mu=_f["posterior"]["mu"].transpose("chain","draw","fit_week").values.reshape(-1,97)
        _mu_interval=np.quantile(_mu,[.025,.975],axis=0)
    _corpus=pg.load_corpus(StudyPath("notebooks/ols_study_50_seed10029")/f"batch_{_record['batch']:02d}.npz")
    _advi=None
    _advi_mu_interval=None
    if scorecard_advi_choice.value is not None and _world<5:
        _method=scorecard_advi_choice.value
        with np.load(StudyPath("notebooks/fast_approximation_first5")/f"world_{_world:02d}_{_method}_recovery.npz",allow_pickle=False) as _f:
            _advi={k:_f[k] for k in _f.files}
        with bayes_xr.open_datatree(StudyPath("notebooks/fast_approximation_first5")/f"world_{_world:02d}_{_method}_posterior.nc",engine="h5netcdf") as _f:
            _advi_mu_interval=np.quantile(_f["posterior"]["mu"].transpose("chain","draw","fit_week").values.reshape(-1,97),[.025,.975],axis=0)
    _indices=[0]+scorecard_geometry["indices"]
    _names=_record["terms"]
    _fit=scorecard_ols_choice.value
    if isinstance(_fit,str) and _fit.startswith("bounded_"):
        with np.load(StudyPath("notebooks/ols_study_50_seed10029_generator_bounds")/f"world_{_world:02d}_fits.npz",allow_pickle=False) as _f:
            _ols={k:_f[k] for k in _f.files}
        _ols_fit_report=next(r for r in bounded_ols_reports if r["world_id"]==_world)["fits"][_fit]
    else:
        _ols_fit_report=_record["fits"][_fit]
    _ols_components=_ols[f"fit_{_fit}_components"]
    _weeks=np.arange(7,104)
    _titles=["Total sales: observed outcome and fitted predictions"]
    for _index in _indices:
        _name=_names[_index]
        if _index==0:
            _titles.append("Intercept (not included in recovery gate)")
        else:
            _allowance=max(.1*np.linalg.norm(_nuts["truth"][:,_index]),.01*np.linalg.norm(_nuts["noiseless"]))
            _nuts_ratio=np.linalg.norm(_nuts["mean"][:,_index]-_nuts["truth"][:,_index])/_allowance
            _ols_ratio=np.linalg.norm(_ols_components[:,_index]-_nuts["truth"][:,_index])/_allowance
            _flat=flatness_status(next(r for r in scorecard_geometry["rows"] if r["Channel"]==_name),scorecard_flatness_cutoff.value/100)
            _titles.append(f"{_name} · error/tolerance: NUTS {_nuts_ratio:.2f}, OLS {_ols_ratio:.2f} · {_flat}")
    _nrows=len(_titles)
    scorecard_world_figure=make_subplots(rows=_nrows,cols=1,shared_xaxes=True,vertical_spacing=.24/_nrows,
        specs=[[{"secondary_y":False}]]+[[{"secondary_y":_names[i].startswith("C")}] for i in _indices],subplot_titles=_titles)
    def _band(row,lower,upper,label,color,legend=True):
        scorecard_world_figure.add_trace(go.Scatter(x=_weeks,y=lower,mode="lines",line={"width":0},showlegend=False,hoverinfo="skip",legendgroup=label),row=row,col=1)
        scorecard_world_figure.add_trace(go.Scatter(x=_weeks,y=upper,mode="lines",line={"width":0},fill="tonexty",fillcolor=color,name=label,legendgroup=label,showlegend=legend,hoverinfo="skip"),row=row,col=1)
    def _line(row,values,label,color,dash="solid",legend=True,secondary=False):
        scorecard_world_figure.add_trace(go.Scatter(x=_weeks,y=values,mode="lines",line={"color":color,"width":1.5,"dash":dash},name=label,legendgroup=label,showlegend=legend),row=row,col=1,secondary_y=secondary)
    _band(1,_nuts["predictive_lower95"],_nuts["predictive_upper95"],"NUTS sales predictive 95%","rgba(255,140,0,.12)")
    _band(1,*_mu_interval,"NUTS expected sales 95%","rgba(255,140,0,.28)")
    _line(1,_nuts["noiseless"],"Truth","black")
    scorecard_world_figure.add_trace(go.Scatter(x=_weeks,y=_nuts["observed"],mode="markers",marker={"size":4,"color":"gray"},name="Observed sales"),row=1,col=1)
    _line(1,_nuts["mean"].sum(axis=1),"NUTS mean","darkorange")
    _line(1,_ols[f"fit_{_fit}_prediction"],"OLS","royalblue","dash")
    if _advi is not None:
        _band(1,*_advi_mu_interval,"ADVI expected sales 95%","rgba(148,103,189,.16)")
        _line(1,_advi["mean"].sum(axis=1),"ADVI mean","#9467bd","dot")
    _spend_legend=True
    for _row,_index in enumerate(_indices,start=2):
        _name=_names[_index]
        _band(_row,_nuts["lower95"][:,_index],_nuts["upper95"][:,_index],"NUTS contribution 95%","rgba(255,140,0,.22)",legend=_row==2)
        if _advi is not None:
            _band(_row,_advi["lower95"][:,_index],_advi["upper95"][:,_index],"ADVI contribution 95%","rgba(148,103,189,.16)",legend=_row==2)
            _line(_row,_advi["mean"][:,_index],"ADVI mean","#9467bd","dot",legend=False)
        _line(_row,_nuts["truth"][:,_index],"Truth","black",legend=False)
        _line(_row,_nuts["mean"][:,_index],"NUTS mean","darkorange",legend=False)
        _line(_row,_ols_components[:,_index],"OLS","royalblue","dash",legend=False)
        if _name.startswith("C"):
            _spend=_corpus["treatment_raw"][_record["local_world"],7:,int(_name[1:])-1]
            _line(_row,_spend,"Raw spend (right axis)","#888888","dot",legend=_spend_legend,secondary=True)
            _spend_legend=False
            scorecard_world_figure.update_yaxes(title_text="Raw spend",showgrid=False,rangemode="tozero",row=_row,col=1,secondary_y=True)
        scorecard_world_figure.update_yaxes(title_text="Contribution",row=_row,col=1,secondary_y=False)
    scorecard_world_figure.update_yaxes(title_text="Sales",row=1,col=1)
    scorecard_world_figure.update_xaxes(showticklabels=True,title_text="Week")
    scorecard_world_figure.update_layout(height=320*_nrows+160,template="plotly_white",title=f"World {_world+1}: all nonzero contributions",margin={"t":100,"b":140},legend={"orientation":"h","y":-.012,"yanchor":"top"})
    _gate=next(r for r in _report["gates"] if r["relative"]==.1 and r["absolute"]==.01)
    scorecard_status_text=(f"**NUTS diagnostics: {'PASS' if _report['diagnostics_pass'] else 'UNRESOLVED — posterior summaries provisional'}.** "
        f"Divergences: {_report['divergences']}; max R-hat: {_report['max_rhat']:.3f}. "
        f"**OLS converged:** {_ols_fit_report['success']}.\n\n"
        f"NUTS point recovery at 10% / 1%: {_gate['error_gates_pass']}; failing channels: {', '.join(_gate['failing_channels']) or 'none'}. "
        "Panel error/tolerance ≤1 passes the point-error gate; convergence/diagnostic validity is separate. "
        "Black = truth, orange = NUTS, blue dashed = OLS; raw spend uses a separate right-hand scale. "
        "All intervals are 95% pointwise equal-tail intervals. Expected-sales bands exclude observation noise; predictive bands include it. "
        "These are in-sample fits, not held-out forecasts. "
        +("ADVI is a potentially unreliable approximation, not a convergence-certified posterior. " if _advi is not None else "")
        +("ADVI unavailable for this world; only worlds 1–5 were fitted." if scorecard_advi_choice.value is not None and _world>=5 else ""))
    mo.vstack([mo.md(scorecard_status_text),mo.ui.plotly(scorecard_world_figure)])

    return


@app.cell(hide_code=True)
def total_media_recovery_view(
    StudyPath,
    bayesian_world_picker,
    go,
    mo,
    np,
    total_media_primary,
):
    with np.load(StudyPath("notebooks/total_media_recovery")/f"world_{bayesian_world_picker.value:02d}_total_media.npz",allow_pickle=False) as _f:
        _arrays={k:_f[k] for k in _f.files}
    _rows=[r for r in total_media_primary if r["World"]==bayesian_world_picker.value+1]
    _fig=go.Figure();_weeks=np.arange(7,104)
    _fig.add_trace(go.Scatter(x=_weeks,y=_arrays["nuts_lower95"],mode="lines",line={"width":0},showlegend=False,hoverinfo="skip"))
    _fig.add_trace(go.Scatter(x=_weeks,y=_arrays["nuts_upper95"],mode="lines",line={"width":0},fill="tonexty",fillcolor="rgba(255,140,0,.22)",name="NUTS total-media 95% interval"))
    for _key,_label,_color,_dash in (("truth","True total media","black","solid"),("nuts_mean","NUTS posterior mean","darkorange","solid"),("ols_mean","Bounded estimated-transform OLS","royalblue","dash")):
        _fig.add_trace(go.Scatter(x=_weeks,y=_arrays[_key],mode="lines",name=_label,line={"color":_color,"dash":_dash}))
    _fig.update_layout(height=500,template="plotly_white",title=f"World {bayesian_world_picker.value+1}: total media contribution",xaxis_title="Week",yaxis_title="Contribution",legend={"orientation":"h","y":-.2,"yanchor":"top"},margin={"b":110})
    mo.vstack([mo.md("### Total media contribution\nAll media summed within each posterior draw; excludes baseline, controls and observation noise. **Always shows bounded estimated-transform OLS**, independently of the main scorecard selector. See computational-validity flags before interpreting the results."),mo.ui.plotly(_fig),mo.ui.table(_rows,selection=None)])

    return


@app.cell(hide_code=True)
def mcmc_channel_baseline_view(
    bayesian_world_picker,
    go,
    mcmc_channel_difficulty,
    mo,
):
    _rows=[r for r in mcmc_channel_difficulty if r["World"]==bayesian_world_picker.value+1]
    _rows=sorted(_rows,key=lambda r:r["Intercept posterior correlation"])
    _fig=go.Figure(go.Bar(x=[r["Intercept posterior correlation"] for r in _rows],y=[r["Channel"] for r in _rows],orientation="h",
        marker={"color":[r["Point error / tolerance"] for r in _rows],"colorscale":"RdYlGn_r","cmin":0,"cmax":3,"colorbar":{"title":"Error / tolerance"}},
        customdata=[[r["True CV"],r["Point error / tolerance"]] for r in _rows],hovertemplate="%{y}<br>Posterior r %{x:.3f}<br>True CV %{customdata[0]:.1%}<br>Error/tolerance %{customdata[1]:.2f}<extra></extra>"))
    _fig.add_vline(x=-.8,line_dash="dash",annotation_text="Strong negative cutoff")
    _fig.update_layout(height=max(300,35*len(_rows)+130),template="plotly_white",title=f"World {bayesian_world_picker.value+1}: media-level/intercept posterior correlation",xaxis={"range":[-1,1],"title":"Posterior correlation"},yaxis={"autorange":"reversed"})
    mo.vstack([mo.md(f"### Individual channels versus baseline\n**MCMC diagnostics pass: {_rows[0]['MCMC valid']}.** If false, these summaries are provisional. Bars correlate each channel's time-average contribution with the intercept across posterior draws. Color clips at error/tolerance =3; hover shows the actual value."),mo.ui.plotly(_fig),mo.ui.table(_rows,selection=None)])

    return


@app.cell(hide_code=True)
def correlated_pair_sum_view(
    StudyPath,
    bayesian_world_picker,
    go,
    mo,
    np,
    pair_primary_rows,
):
    _rows=[r for r in pair_primary_rows if r["World"]==bayesian_world_picker.value+1]
    pair_sum_view=mo.md("No eligible positively correlated pair of nonzero, varying media in this world.")
    if _rows:
        _nuts=next(r for r in _rows if r["Method"]=="NUTS")
        with np.load(StudyPath("notebooks/correlated_media_pair_recovery")/f"world_{bayesian_world_picker.value:02d}_pair.npz",allow_pickle=False) as _f:
            _pair={k:_f[k] for k in _f.files}
        _fig=go.Figure()
        _weeks=np.arange(7,104)
        _fig.add_trace(go.Scatter(x=_weeks,y=_pair["nuts_lower95"],mode="lines",line={"width":0},showlegend=False,hoverinfo="skip"))
        _fig.add_trace(go.Scatter(x=_weeks,y=_pair["nuts_upper95"],mode="lines",line={"width":0},fill="tonexty",fillcolor="rgba(255,140,0,.22)",name="NUTS sum: 95% pointwise interval"))
        for _key,_label,_color,_dash in (("truth","True sum","black","solid"),("nuts_mean","NUTS posterior mean","darkorange","solid"),("ols_mean","Bounded estimated-transform OLS","royalblue","dash")):
            _fig.add_trace(go.Scatter(x=_weeks,y=_pair[_key],mode="lines",name=_label,line={"color":_color,"dash":_dash}))
        _fig.update_layout(height=480,template="plotly_white",title=f"World {bayesian_world_picker.value+1}: {_nuts['Pair']} · true r={_nuts['True correlation']:.3f}",xaxis_title="Week",yaxis_title="Summed contribution",legend={"orientation":"h","y":-.2,"yanchor":"top"},margin={"b":110})
        pair_sum_view=mo.vstack([mo.md("### Highest-correlated media pair: summed recovery\nAlways compares **bounded estimated-transform OLS**, independently of the main scorecard OLS selector. MCMC intervals are computed after summing within each joint draw, not by adding marginal interval endpoints. **Check computational-validity flags before interpreting either fit.**"),mo.ui.plotly(_fig),mo.ui.table(_rows,selection=None)])
    pair_sum_view

    return


@app.cell
def bayesian_contribution_selection(
    StudyPath,
    bayesian_reports,
    bayesian_world_picker,
    mo,
    np,
):
    bayesian_view_report=next(r for r in bayesian_reports if r["world_id"]==bayesian_world_picker.value)
    with np.load(StudyPath("notebooks/bayesian_mmm_draw_diag")/f"world_{bayesian_world_picker.value:02d}_recovery.npz",allow_pickle=False) as _saved:
        bayesian_view_arrays={name:_saved[name] for name in _saved.files}
    bayesian_component_picker=mo.ui.dropdown(
        options={name:index for index,name in enumerate(bayesian_view_report["terms"]) if name=="Intercept" or np.any(bayesian_view_arrays["truth"][:,index]!=0)},
        value=next((name for index,name in enumerate(bayesian_view_report["terms"]) if name!="Intercept" and np.any(bayesian_view_arrays["truth"][:,index]!=0)), "Intercept"),
        label="Posterior contribution",
    )
    mo.vstack([mo.md(f"**Diagnostics pass:** {bayesian_view_report['diagnostics_pass']} · "
        f"**Divergences:** {bayesian_view_report['divergences']} · **Maximum-depth hits:** {bayesian_view_report['maxdepth_hits']}\n\n"
        "Shading shows a **95% pointwise equal-tail posterior interval** for this contribution, not observation noise. "
        "A wide interval containing truth does not mean precise recovery."),bayesian_component_picker])
    return (
        bayesian_component_picker,
        bayesian_view_arrays,
        bayesian_view_report,
    )


@app.cell
def bayesian_contribution_plot(
    StudyPath,
    bayesian_component_picker,
    bayesian_view_arrays,
    bayesian_view_report,
    bayesian_world_picker,
    go,
    mo,
    np,
    pg,
    plot_contribution_recovery,
    saved_study_records,
):
    _term_name=bayesian_view_report["terms"][bayesian_component_picker.value]
    _selected_spend=None
    if _term_name.startswith("C") and _term_name[1:].isdigit():
        _record=saved_study_records[bayesian_world_picker.value]
        _corpus=pg.load_corpus(StudyPath("notebooks/ols_study_50_seed10029")/f"batch_{_record['batch']:02d}.npz")
        _selected_spend=_corpus["treatment_raw"][_record["local_world"],7:,int(_term_name[1:])-1].astype(float)
    bayesian_contribution_figure=plot_contribution_recovery(
        bayesian_view_arrays["truth"],bayesian_view_arrays["mean"],bayesian_view_arrays["observed"],
        bayesian_view_arrays["mean"].sum(axis=1),7,bayesian_view_report["terms"],bayesian_component_picker.value,
        f"MCMC study world {bayesian_world_picker.value+1} · posterior means (Nutpie draw_diag)",
        media_spend=_selected_spend,
    )
    bayesian_contribution_figure.data[2].name="Posterior mean outcome"
    bayesian_contribution_figure.data[4].name="Posterior mean contribution"
    bayesian_contribution_figure.add_trace(go.Scatter(x=np.arange(7,104),
        y=bayesian_view_arrays["lower95"][:,bayesian_component_picker.value],mode="lines",line={"width":0},
        showlegend=False,hoverinfo="skip",legend="legend2"),row=2,col=1)
    bayesian_contribution_figure.add_trace(go.Scatter(x=np.arange(7,104),
        y=bayesian_view_arrays["upper95"][:,bayesian_component_picker.value],mode="lines",line={"width":0},
        fill="tonexty",fillcolor="rgba(255,140,0,0.2)",name="95% posterior interval",legend="legend2"),row=2,col=1)
    _variability_note=""
    if _selected_spend is not None:
        _true=bayesian_view_arrays["truth"][:,bayesian_component_picker.value]
        _mean=bayesian_view_arrays["mean"][:,bayesian_component_picker.value]
        _spend_cv=float(np.std(_selected_spend)/np.mean(_selected_spend)) if np.mean(_selected_spend)>0 else None
        _true_cv=float(np.std(_true)/abs(np.mean(_true))) if abs(np.mean(_true))>0 else None
        _mean_cv=float(np.std(_mean)/abs(np.mean(_mean))) if abs(np.mean(_mean))>0 else None
        _cv_text=lambda v: f"{v:.1%}" if v is not None else "undefined (zero mean)"
        _variability_note=(f"**Temporal variation, weeks 7–103:** raw-spend CV {_cv_text(_spend_cv)}; "
            f"true-contribution CV {_cv_text(_true_cv)}; posterior-mean-contribution CV {_cv_text(_mean_cv)}. "
            "CV is time-series SD / absolute mean, not posterior uncertainty. "
            "Spend is **before adstock/saturation** and uses a separate right-hand scale: compare timing, not line heights. "
            "Click its legend entry to hide/show it.")
    mo.accordion({"Legacy single-channel contribution plot":mo.vstack([mo.ui.plotly(bayesian_contribution_figure),mo.md(_variability_note)])})
    return


@app.cell
def bayesian_trace_view(
    StudyPath,
    bayes_xr,
    bayesian_world_picker,
    go,
    make_subplots,
    mo,
    np,
    qualitative,
):
    with bayes_xr.open_datatree(StudyPath("notebooks/bayesian_mmm_draw_diag")/f"world_{bayesian_world_picker.value:02d}_posterior.nc",engine="h5netcdf") as _trace_file:
        bayesian_trace_values = {
            "intercept":_trace_file["posterior"]["intercept"].values.copy(),
            "sigma":_trace_file["posterior"]["sigma"].values.copy(),
        }
    bayesian_trace_figure=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.2,subplot_titles=("Intercept chains","Noise SD chains"))
    for _row,(_name,_values) in enumerate(bayesian_trace_values.items(),start=1):
        for _chain in range(4):
            bayesian_trace_figure.add_trace(go.Scatter(x=np.arange(_values.shape[1]),y=_values[_chain],mode="lines",
                name=f"Chain {_chain+1}",legend="legend" if _row==1 else "legend2",line={"width":0.6,"color":qualitative.Plotly[_chain]}),row=_row,col=1)
    bayesian_trace_figure.update_layout(height=650,template="plotly_white",title=f"Study world {bayesian_world_picker.value+1}: post-warmup chains",
        legend={"orientation":"h","y":0.55,"yanchor":"top"},legend2={"orientation":"h","y":-0.12,"yanchor":"top"})
    mo.accordion({"Optional legacy MCMC trace plots":mo.ui.plotly(bayesian_trace_figure)})
    return


@app.cell
def approximation_imports(mo):
    import pymc_extras as approximation_pmx
    import warnings as approximation_warnings
    from importlib.metadata import version as approximation_version
    from pymc_extras.inference.pathfinder.lbfgs import LBFGSConfig as ApproximationLBFGSConfig
    mo.md("### Fast approximations: Pathfinder versus full-rank ADVI\n\nSame five worlds/model/priors as NUTS. Pathfinder: four paths, PSIS, 1,000 output draws from 4,000 candidates. The initial 4,000-output pilot is retained separately. ADVI: two independent seeds, 20,000 iterations each, 4,000 draws per fit. No NUTS-informed initialization. Approximation draws are not MCMC chains; no R-hat or MCMC ESS is reported for them.")
    return (
        ApproximationLBFGSConfig,
        approximation_pmx,
        approximation_version,
        approximation_warnings,
    )


@app.cell
def fast_approximation_functions(
    ApproximationLBFGSConfig,
    StudyPath,
    approximation_pmx,
    approximation_version,
    approximation_warnings,
    bayes_az,
    bayes_pm,
    bayesian_reports,
    bayesian_world_data,
    build_bayesian_mmm,
    channel_gate_metrics,
    constant_baseline_cfg,
    np,
    perf_counter,
    posterior_component_draws,
    prepare_ols_case,
    relative_component_error,
    replace,
    saved_study_records,
    study_hashlib,
    study_inspect,
    study_json,
    study_json_safe,
    write_study_json,
):
    def score_fast_approximation(idata, record, data, corpus, directory, label):
        components,mu=posterior_component_draws(idata,data,record)
        assert np.isfinite(components).all()
        mean=components.mean(axis=0)
        low,high=np.quantile(components,[0.025,0.975],axis=0)
        case=prepare_ols_case(corpus,record["local_world"],replace(constant_baseline_cfg,seed=record["seed"]))
        n=record["media"]
        with np.load(StudyPath("notebooks/bayesian_mmm_draw_diag")/f"world_{record['world_id']:02d}_recovery.npz") as reference:
            reference_mean=reference["mean"].copy()
            reference_width=reference["upper95"]-reference["lower95"]
            reference_low=reference["lower95"].copy(); reference_high=reference["upper95"].copy()
        gates=[]
        for relative in (0.05,0.1):
            for absolute in (0.005,0.01):
                error,norm,allowed,passed,ratio=channel_gate_metrics(case["truth"][:,1:],mean[:,1:],np.linalg.norm(case["noiseless"][7:]),relative,absolute)
                gates.append({"relative":relative,"absolute":absolute,"point_error_gates_pass":bool(passed.all()),
                    "media_point_pass":bool(passed[:n].all()),"control_point_pass":bool(passed[n:].all()),
                    "failing_channels":[name for name,ok in zip(case["names"][1:],passed,strict=True) if not ok]})
        widths=high-low
        # Ratios are undefined where the NUTS interval has zero width (e.g. zero input).
        reference_scales=np.where(reference_width[:,1:]>1e-12,reference_width[:,1:],np.nan)
        summary={"world_id":record["world_id"],"method":label,"gates":gates,
            "nuts_reference_diagnostics_pass":bayesian_reports[record["world_id"]]["diagnostics_pass"],
            "media_relative_error":relative_component_error(mean[:,1:1+n],case["truth"][:,1:1+n]),
            "control_relative_error":relative_component_error(mean[:,1+n:],case["truth"][:,1+n:]),
            "noiseless_rmse":float(np.sqrt(np.mean((mu.mean(axis=0)-case["noiseless"][7:])**2))),
            "media_95pct_pointwise_truth_coverage":float(np.mean((case["truth"][:,1:1+n]>=low[:,1:1+n])&(case["truth"][:,1:1+n]<=high[:,1:1+n]))),
            "control_95pct_pointwise_truth_coverage":float(np.mean((case["truth"][:,1+n:]>=low[:,1+n:])&(case["truth"][:,1+n:]<=high[:,1+n:]))),
            "median_media_interval_width_ratio_to_nuts":float(np.nanmedian(widths[:,1:1+n]/reference_scales[:,:n])),
            "median_control_interval_width_ratio_to_nuts":float(np.nanmedian(widths[:,1+n:]/reference_scales[:,n:])),
            "median_abs_mean_gap_in_nuts_interval_widths":float(np.nanmedian(np.abs(mean[:,1:]-reference_mean[:,1:])/reference_scales)),
            "p95_abs_mean_gap_in_nuts_interval_widths":float(np.nanquantile(np.abs(mean[:,1:]-reference_mean[:,1:])/reference_scales,0.95)),
            "median_endpoint_gap_in_nuts_interval_widths":float(np.nanmedian((np.abs(low[:,1:]-reference_low[:,1:])+np.abs(high[:,1:]-reference_high[:,1:]))/(2*reference_scales))),
            "valid_media_interval_comparisons":int(np.isfinite(reference_scales[:,:n]).sum()),"valid_control_interval_comparisons":int(np.isfinite(reference_scales[:,n:]).sum()),"terms":case["names"]}
        np.savez_compressed(directory/f"world_{record['world_id']:02d}_{label}_recovery.npz",truth=case["truth"],mean=mean,lower95=low,upper95=high)
        return study_json_safe(summary)


    def run_fast_approximation_study():
        directory=StudyPath("notebooks/fast_approximation_first5")
        directory.mkdir(exist_ok=True,parents=True)
        manifest={"source_manifest":study_json.loads(StudyPath("notebooks/ols_study_50_seed10029/manifest.json").read_text()),"world_ids":list(range(5)),"pymc":bayes_pm.__version__,"pytensor":approximation_version("pytensor"),
            "pymc_extras":approximation_pmx.__version__,"advi_draws":4000,"pathfinder_draws":1000,"advi_iterations":20000,
            "advi_replicates":2,"advi_optimizer":"PyMC default adagrad_window","advi_obj_n_mc":1,
            "pathfinder_paths":4,"pathfinder_draws_per_path":1000,"pathfinder_maxiter":1000,
            "pathfinder_parallel":True,"pathfinder_cores":4,"pathfinder_importance_sampling":"psis",
            "pathfinder_jacobian_correction":True,"pathfinder_vectorize_postprocessing":False,"pathfinder_seeds":[81001+i for i in range(5)],
            "advi_seeds":[[82001+10*i+j for j in range(2)] for i in range(5)],
            "model_sha256":study_hashlib.sha256((study_inspect.getsource(build_bayesian_mmm)+study_inspect.getsource(bayesian_world_data)).encode()).hexdigest()}
        if (directory/"manifest.json").exists():
            assert study_json.loads((directory/"manifest.json").read_text())==manifest,"Approximation manifest mismatch"
        else:
            if any(directory.iterdir()):raise ValueError("Nonempty approximation directory without manifest")
            write_study_json(directory/"manifest.json",manifest)
        all_reports=[]
        for record in saved_study_records[:5]:
            world=record["world_id"]
            data,corpus=bayesian_world_data(record)
            model=build_bayesian_mmm(data)
            for label in ("pathfinder_1000","fullrank_advi_1","fullrank_advi_2"):
                prefix=directory/f"world_{world:02d}_{label}"
                posterior_path=prefix.with_name(prefix.name+"_posterior.nc")
                metadata_path=prefix.with_name(prefix.name+"_metadata.json")
                if posterior_path.exists():
                    idata=bayes_az.from_netcdf(posterior_path,engine="h5netcdf")
                    metadata=study_json.loads(metadata_path.read_text())
                else:
                    print(f"Starting {label}, world {world+1}/5",flush=True)
                    started=perf_counter()
                    metadata={"world_id":world,"method":label}
                    with approximation_warnings.catch_warnings(record=True) as caught:
                        approximation_warnings.simplefilter("always")
                        if label=="pathfinder_1000":
                            idata=approximation_pmx.fit_pathfinder(model=model,num_paths=4,num_draws=1000,
                                num_draws_per_path=1000,num_elbo_draws=10,
                                lbfgs_config=ApproximationLBFGSConfig(maxiter=1000),importance_sampling="psis",
                                parallel=True,cores=4,blas_cores=4,progressbar=False,random_seed=81001+world,
                                jacobian_correction=True,vectorize_postprocessing=False)
                            stats=idata["pathfinder"]
                            metadata.update(compile_seconds=float(stats["compile_time"]),fit_seconds=float(stats["compute_time"]),
                                pareto_k=float(stats["pareto_k"]) if "pareto_k" in stats else None,
                                successful_paths=int(stats["num_successful_paths"]) if "num_successful_paths" in stats else 0,
                                pathfinder_warnings=stats.attrs.get("warnings",[]))
                        else:
                            replicate=int(label[-1])-1
                            seed=82001+10*world+replicate
                            with model:
                                inference=bayes_pm.FullRankADVI(random_seed=seed)
                                compilation_start=perf_counter()
                                inference.fit(n=0,score=True,progressbar=False,obj_n_mc=1)
                                metadata["compile_seconds"]=perf_counter()-compilation_start
                                fit_start=perf_counter()
                                inference.refine(5000,progressbar=False)
                                checkpoint={"iteration":5000,"seconds":perf_counter()-fit_start,
                                    "last_1000_mean_loss":float(np.mean(inference.hist[-1000:]))}
                                inference.refine(15000,progressbar=False)
                                metadata["fit_seconds"]=perf_counter()-fit_start
                                losses=inference.hist.copy()
                                # Persist fitted distribution and full loss before sampling/reporting.
                                np.savez_compressed(prefix.with_name(prefix.name+"_fit.npz"),loss=losses,
                                    **{f"parameter_{i}":p.get_value() for i,p in enumerate(inference.approx.params)})
                                metadata.update(iterations=len(losses),checkpoint_5000=checkpoint,
                                    last_1000_mean_loss=float(np.mean(losses[-1000:])),
                                    previous_1000_mean_loss=float(np.mean(losses[-2000:-1000])),
                                    finite_losses=bool(np.isfinite(losses).all()))
                                draw_start=perf_counter()
                                idata=inference.approx.sample(draws=4000,random_seed=seed+1000)
                                metadata["draw_seconds"]=perf_counter()-draw_start
                        metadata["warnings"]=[str(w.message) for w in caught]
                    metadata["total_seconds_before_persistence"]=perf_counter()-started
                    metadata["other_seconds"]=metadata["total_seconds_before_persistence"]-metadata["compile_seconds"]-metadata["fit_seconds"]
                    write_study_json(metadata_path,study_json_safe(metadata))
                    idata.attrs.update(study_world_id=world,inference_method=label)
                    idata.to_netcdf(posterior_path,engine="h5netcdf")
                report=score_fast_approximation(idata,record,data,corpus,directory,label)
                report.update(metadata)
                write_study_json(prefix.with_name(prefix.name+"_report.json"),study_json_safe(report))
                all_reports.append(report)
                write_study_json(directory/"reports.json",study_json_safe(all_reports))
                print(f"Finished {label}, world {world+1}: total={report['total_seconds_before_persistence']:.1f}s, media error={report['media_relative_error']:.3f}, interval ratio={report['median_media_interval_width_ratio_to_nuts']:.3f}",flush=True)
        return all_reports

    return run_fast_approximation_study, score_fast_approximation


@app.cell
def fast_approximation_control(mo):
    approximation_run_button=mo.ui.run_button(label="Run / resume Pathfinder and full-rank ADVI on five saved worlds")
    approximation_run_button
    return (approximation_run_button,)


@app.cell
def run_fast_approximation_study_cell(
    approximation_run_button,
    mo,
    run_fast_approximation_study,
):
    mo.stop(not approximation_run_button.value)
    approximation_reports=run_fast_approximation_study()
    return (approximation_reports,)


@app.cell
def fast_approximation_report(approximation_reports, mo):
    approximation_summary=[{
        "World":r["world_id"]+1,"Method":r["method"],"Compile s":r["compile_seconds"],"Fit s":r["fit_seconds"],
        "Total s":r["total_seconds_before_persistence"],"Media error":r["media_relative_error"],"Control error":r["control_relative_error"],
        "Media interval width / NUTS":r["median_media_interval_width_ratio_to_nuts"],
        "Control interval width / NUTS":r["median_control_interval_width_ratio_to_nuts"],
        "Media truth coverage":r["media_95pct_pointwise_truth_coverage"],"Pareto k":r.get("pareto_k"),
        "All point gates at 10% / 1%":r["gates"][-1]["point_error_gates_pass"],
        "NUTS reference passes":r["nuts_reference_diagnostics_pass"],
    } for r in approximation_reports]
    mo.vstack([mo.md("### Fast approximation comparison\n\nBoth ADVI seeds are reported separately, not pooled or selected by truth. Point-error passes are **not** certificates of valid inference. Check the NUTS diagnostic flag for each world before treating it as a reference. Timings describe these public-API calls; Nutpie reused its compilation whereas these approximation calls may compile each fit."),
               mo.ui.table(approximation_summary,selection=None)])
    return


@app.cell
def approximation_reliability(
    StudyPath,
    approximation_reports,
    bayes_xr,
    mo,
    np,
    write_study_json,
):
    approximation_diagnostic_rows=[]
    for _r in approximation_reports:
        _prefix=StudyPath("notebooks/fast_approximation_first5")/f"world_{_r['world_id']:02d}_{_r['method']}"
        if _r["method"].startswith("pathfinder"):
            with bayes_xr.open_datatree(_prefix.with_name(_prefix.name+"_posterior.nc"),engine="h5netcdf") as _tree:
                _unique=len(np.unique(_tree["posterior"]["intercept"].values))
                _counts={str(_status):int(_count) for _status,_count in zip(_tree["lbfgs"]["status"].values,_tree["lbfgs"]["status_counts"].values,strict=True)}
                _iterations=_tree["lbfgs"]["niter"].values.tolist()
            approximation_diagnostic_rows.append({"World":_r["world_id"]+1,"Method":_r["method"],
                "Pareto k":_r["pareto_k"],"Importance reliability flag":"unreliable" if _r["pareto_k"]>0.7 else "no k flag",
                "Unique output intercepts":_unique,"LBFGS status counts":_counts,"LBFGS iterations":_iterations,
                "Warnings":_r["pathfinder_warnings"]})
        else:
            approximation_diagnostic_rows.append({"World":_r["world_id"]+1,"Method":_r["method"],
                "Last 1000 loss":_r["last_1000_mean_loss"],"Previous 1000 loss":_r["previous_1000_mean_loss"],
                "Optimizer convergence":"not established; fixed 20,000-step budget","Finite losses":_r["finite_losses"]})
    write_study_json(StudyPath("notebooks/fast_approximation_first5/diagnostic_summary.json"),approximation_diagnostic_rows)
    mo.vstack([mo.md("### Approximation reliability\n\nAll five Pathfinder fits have large importance-weight Pareto k (this is **not LOO k**). ADVI's objective is still decreasing at the fixed budget, so its posterior accuracy is unresolved. Neither a completed optimization nor two similar ADVI seeds establishes posterior accuracy.\n\n**Timing caveats:** Pathfinder used four workers. API timing splits can include deferred compilation in computation/postprocessing. These reruns used an already-warm session; timings are not controlled cold-start benchmarks. No exact speedup or equal-accuracy benchmark is claimed."),mo.ui.table(approximation_diagnostic_rows,selection=None)])
    return


@app.cell
def approximation_loss_plot(StudyPath, go, make_subplots, mo, np, qualitative):
    mo.stop(not StudyPath("notebooks/fast_approximation_first5/reports.json").exists())
    approximation_loss_figure=make_subplots(rows=3,cols=2,subplot_titles=[f"World {i+1}" for i in range(5)]+[""],vertical_spacing=0.13)
    for _world in range(5):
        for _rep in (1,2):
            with np.load(StudyPath("notebooks/fast_approximation_first5")/f"world_{_world:02d}_fullrank_advi_{_rep}_fit.npz") as _fit:
                _loss=_fit["loss"]
            _means=_loss.reshape(-1,500).mean(axis=1)
            approximation_loss_figure.add_trace(go.Scatter(x=np.arange(500,len(_loss)+1,500),y=_means,mode="lines",name=f"ADVI seed {_rep}",
                legendgroup=f"seed{_rep}",showlegend=_world==0,line={"color":qualitative.Plotly[_rep-1]}),row=_world//2+1,col=_world%2+1)
    approximation_loss_figure.update_yaxes(type="log",title_text="Mean loss (500 steps)")
    approximation_loss_figure.update_xaxes(title_text="Iteration")
    approximation_loss_figure.update_layout(height=850,template="plotly_white",title="Full-rank ADVI loss: both seeds, all five worlds",
        legend={"orientation":"h","y":-0.1,"yanchor":"top"})
    mo.ui.plotly(approximation_loss_figure)
    return


@app.cell
def approximation_view_selector(mo):
    approximation_view_picker=mo.ui.dropdown(options={"Pathfinder (1,000 outputs)":"pathfinder_1000","Full-rank ADVI seed 1":"fullrank_advi_1","Full-rank ADVI seed 2":"fullrank_advi_2"},value="Pathfinder (1,000 outputs)",label="Approximation overlay")
    mo.vstack([mo.md("### Compare posterior contribution intervals\n\nUses the **MCMC world and contribution selectors above**. Narrow approximation bands should not be interpreted as stronger identification when the approximation fails."),approximation_view_picker])
    return (approximation_view_picker,)


@app.cell
def approximation_uncertainty_overlay(
    StudyPath,
    approximation_reports,
    approximation_view_picker,
    bayesian_component_picker,
    bayesian_view_arrays,
    bayesian_view_report,
    bayesian_world_picker,
    go,
    mo,
    np,
):
    mo.stop(not approximation_reports)
    mo.stop(bayesian_world_picker.value>=5,mo.md("Pathfinder/ADVI were fitted only on worlds 1–5; NUTS is available for all 50."))
    with np.load(StudyPath("notebooks/fast_approximation_first5")/f"world_{bayesian_world_picker.value:02d}_{approximation_view_picker.value}_recovery.npz") as _saved:
        approximation_view_arrays={key:_saved[key] for key in _saved.files}
    approximation_overlay_figure=go.Figure()
    for _label,_arrays,_color,_fill in (("NUTS",bayesian_view_arrays,"#636efa","rgba(99,110,250,0.18)"),
            (approximation_view_picker.value,approximation_view_arrays,"#ef553b","rgba(239,85,59,0.18)")):
        approximation_overlay_figure.add_trace(go.Scatter(x=np.arange(7,104),y=_arrays["lower95"][:,bayesian_component_picker.value],
            mode="lines",line={"width":0},showlegend=False,hoverinfo="skip"))
        approximation_overlay_figure.add_trace(go.Scatter(x=np.arange(7,104),y=_arrays["upper95"][:,bayesian_component_picker.value],
            mode="lines",line={"width":0},fill="tonexty",fillcolor=_fill,name=f"{_label} 95% ETI"))
        approximation_overlay_figure.add_trace(go.Scatter(x=np.arange(7,104),y=_arrays["mean"][:,bayesian_component_picker.value],
            mode="lines",line={"color":_color},name=f"{_label} mean"))
    approximation_overlay_figure.add_trace(go.Scatter(x=np.arange(7,104),y=bayesian_view_arrays["truth"][:,bayesian_component_picker.value],
        mode="lines",line={"color":"black","dash":"dash"},name="Truth"))
    approximation_overlay_figure.update_layout(height=500,template="plotly_white",
        title=f"World {bayesian_world_picker.value+1} · {bayesian_view_report['terms'][bayesian_component_picker.value]}",
        legend={"orientation":"h","y":-0.15,"yanchor":"top"},xaxis_title="Week",yaxis_title="Contribution")
    mo.ui.plotly(approximation_overlay_figure)
    return


@app.cell
def approximation_saved_report(
    StudyPath,
    approximation_reports,
    bayesian_reports,
    mo,
    run_fast_approximation_study,
    score_fast_approximation,
    study_inspect,
):
    _reference_failures=[r["world_id"]+1 for r in bayesian_reports if r["world_id"]<5 and not r["diagnostics_pass"]]
    _approximate_point_passes=sum(r["gates"][-1]["point_error_gates_pass"] for r in approximation_reports)
    approximation_report_text=f"""# Current five-world approximation analysis

    The canonical 50-world corpus was regenerated in place with effect counts uniform over 1..N active media and 0..M active controls. Old fit artifacts were invalidated. This report uses the first five corrected datasets.

    Oracle-graph comparison: all fits know true zero/nonzero effect identities. Bayesian fits additionally condition on generator saturation anchors and use the generator continuous priors: uniform reference effects, baseline, adstock and relative noise, and log-uniform MM multipliers. No realized nonzero coefficient or shape is supplied. This is conditional prior matching, not an unprivileged inference benchmark. Coverage over all active inputs includes known zeros.

    - Pathfinder: four parallel paths, 1,000 candidates/path, 1,000 output draws, requested PSIS, Jacobian correction; scan-based postprocessing.
    - Full-rank ADVI: two seeds/world, 20,000 iterations each, default optimizer, 4,000 output draws. Both seeds retained separately.
    - Nutpie: draw_diag, four chains, 1,000 tune and 1,000 retained draws/chain. Worlds failing the diagnostic screen: {_reference_failures}. These are provisional references.
    - Approximation point-gate passes at 10% relative / 1% outcome: {_approximate_point_passes}/{len(approximation_reports)}. This does not certify approximation accuracy.

    Pathfinder's large importance-weight k is a reliability warning, not LOO. ADVI loss curves must be inspected; a fixed iteration budget does not establish convergence. Compare posterior means AND interval widths, not only point recovery. Timings include this session's existing compilation cache and are not controlled cold-start comparisons. Pathfinder uses four workers; NUTS compilation is reused, whereas approximation API calls compile each fit.

    Width ratios exclude reference intervals <=1e-12. Coverage is pointwise, not simultaneous or population calibration.

    | World | Method | Total seconds | Media relative error | Median media interval width / NUTS | Pareto k |
    |---|---|---:|---:|---:|---:|
    """
    for _r in approximation_reports:
        approximation_report_text+=f"| {_r['world_id']+1} | {_r['method']} | {_r['total_seconds_before_persistence']:.2f} | {_r['media_relative_error']:.3f} | {_r['median_media_interval_width_ratio_to_nuts']:.3f} | {_r.get('pareto_k','—')} |\n"
    _approximation_directory=StudyPath("notebooks/fast_approximation_first5")
    (_approximation_directory/"README.md").write_text(approximation_report_text)
    (_approximation_directory/"implementation.py.txt").write_text(study_inspect.getsource(run_fast_approximation_study)+"\n"+study_inspect.getsource(score_fast_approximation))
    mo.md("Current regenerated-data analysis saved under `notebooks/fast_approximation_first5/`. Obsolete results were replaced, not retained as a second experiment.")
    return


@app.cell
def oracle_analysis_validation(
    StudyPath,
    bayes_xr,
    bayesian_reports,
    bayesian_world_data,
    constant_baseline_cfg,
    mo,
    np,
    pg,
    prepare_ols_case,
    saved_study_records,
    transform_media_columns,
    write_study_json,
):
    mo.stop(len(bayesian_reports)<50,mo.md("Full validation runs after all 50 NUTS fits finish."))
    oracle_validation={"ols_fits_checked":0,"inference_outputs_checked":0,"max_scaled_prediction_difference":0.0,"max_scaled_component_difference":0.0}
    for _r in saved_study_records:
        _corpus=pg.load_corpus(StudyPath("notebooks/ols_study_50_seed10029")/f"batch_{_r['batch']:02d}.npz")
        _case=prepare_ols_case(_corpus,_r["local_world"],constant_baseline_cfg)
        _m=_case["media_effect_mask"];_c=_case["control_effect_mask"]
        _nonnull=np.r_[True,_m,_c]
        _media=transform_media_columns(_case["media"][:,_m],_case["alpha"][_m],_case["half"][_m])
        _X=np.column_stack([np.ones(len(_media)),_media,_case["controls"][:,_c]])[7:]
        _xm=_X[:,1:].mean(axis=0);_xs=_X[:,1:].std(axis=0);_xs=np.where(_xs>0,_xs,1.)
        for _folder in ("ols_study_50_seed10029","ols_study_50_seed10029_saturation95"):
            with np.load(StudyPath("notebooks")/_folder/f"world_{_r['world_id']:02d}_fits.npz") as _a:
                for _idx in range(4):
                    assert np.all(_a[f"fit_{_idx}_coefficients"][~_nonnull]==0)
                    assert np.all(_a[f"fit_{_idx}_components"][:,~_nonnull]==0)
                    oracle_validation["ols_fits_checked"]+=1
                if _folder=="ols_study_50_seed10029":
                    for _idx,_y in ((2,_case["observed"][7:]),(3,_case["noiseless"][7:])):
                        _ys=max(float(_y.std()),1e-15);_ym=_y.mean()
                        _coef=np.linalg.lstsq((_X[:,1:]-_xm)/_xs,(_y-_ym)/_ys,rcond=None)[0]*_ys/_xs
                        _coef=np.r_[_ym-_xm@_coef,_coef]
                        oracle_validation["max_scaled_prediction_difference"]=max(oracle_validation["max_scaled_prediction_difference"],float(np.max(np.abs(_X@_coef-_a[f"fit_{_idx}_prediction"]))))
                        oracle_validation["max_scaled_component_difference"]=max(oracle_validation["max_scaled_component_difference"],float(np.max(np.abs(_X*_coef-_a[f"fit_{_idx}_components"][:,_nonnull]))))
    for _r in saved_study_records:
        _data,_corpus=bayesian_world_data(_r)
        _paths=[(StudyPath("notebooks/bayesian_mmm_draw_diag")/f"world_{_r['world_id']:02d}_posterior.nc","outcome_scaled")]
        _paths += [(StudyPath("notebooks/fast_approximation_first5")/f"world_{_r['world_id']:02d}_{method}_posterior.nc","outcome") for method in ("pathfinder_1000","fullrank_advi_1","fullrank_advi_2")] if _r["world_id"]<5 else []
        for _path,_yname in _paths:
            with bayes_xr.open_datatree(_path,engine="h5netcdf") as _tree:
                _p=_tree["posterior"]
                np.testing.assert_allclose(_tree["observed_data"][_yname].values,_data["outcome_data"],rtol=1e-14)
                np.testing.assert_array_equal(_tree["constant_data"]["media_effect_mask"].values,_data["media_effect_mask"])
                np.testing.assert_array_equal(_tree["constant_data"]["control_effect_mask"].values,_data["control_effect_mask"])
                np.testing.assert_array_equal(_tree["constant_data"]["saturation_anchor"].values,_data["saturation_anchor"])
                _beta=_p["beta"].values;_rho=_p["rho"].values
                assert np.all(_beta>=0)
                assert np.all(_beta[...,_data["media_effect_mask"]==0]==0)
                assert np.all(_rho[...,_data["control_effect_mask"]==0]==0)
                assert np.all((_p["amplitude"].values>=0.5)&(_p["amplitude"].values<=2))
                assert np.all((_p["mm_kappa_mult"].values>=0.05)&(_p["mm_kappa_mult"].values<=20))
                assert np.all((_p["sigma_relative"].values>=0.01)&(_p["sigma_relative"].values<=0.028))
                np.testing.assert_allclose(_p["sigma"].values,_p["sigma_relative"].values*np.sqrt((_beta**2).sum(axis=-1)),rtol=1e-12)
                np.testing.assert_allclose(_p["half_saturation"].values,_p["mm_kappa_mult"].values*_data["saturation_anchor"],rtol=1e-12)
                oracle_validation["inference_outputs_checked"]+=1
    write_study_json(StudyPath("notebooks/oracle_validation.json"),oracle_validation)
    write_study_json(StudyPath("notebooks/ols_scaling_check.json"),oracle_validation)
    mo.md(f"**Oracle checks passed:** {oracle_validation['ols_fits_checked']} saved OLS variants (including copied fixed-transform fits), {oracle_validation['inference_outputs_checked']} posterior files. Known-null coefficients/contributions are exactly zero; Bayesian media coefficients are nonnegative; generator supports, noise coupling, anchors, and current outcome bindings verified.")
    return


@app.cell
def nuts_ols_full_study_comparison(
    StudyPath,
    bayesian_reports,
    constrained_channel_scores,
    mo,
    original_channel_scores,
    write_study_json,
):
    nuts_ols_comparison_rows=[]
    for _mode,_scores,_prefix in (("Estimated OLS",original_channel_scores,"1 ·"),
                                 ("Constrained estimated OLS",constrained_channel_scores,"1 ·"),
                                 ("True-transform OLS",original_channel_scores,"3 ·")):
        _lookup={(r["world_id"],r["relative"],r["absolute"]):r for r in _scores if r["fit"].startswith(_prefix)}
        for _report in bayesian_reports:
            for _gate in _report["gates"]:
                _ols=_lookup[(_report["world_id"],_gate["relative"],_gate["absolute"])]
                _healthy=_report["diagnostics_pass"]
                _opass=_ols["joint_pass"]
                _npass=_gate["error_gates_pass"]
                _classification=("NUTS unresolved" if not _healthy else
                    "Both recover" if _opass and _npass else
                    "NUTS only" if _npass else "OLS only" if _opass else "Neither recovers")
                nuts_ols_comparison_rows.append({"OLS method":_mode,"World":_report["world_id"]+1,
                    "Relative tolerance":_gate["relative"],"Outcome tolerance":_gate["absolute"],
                    "OLS converged":_ols["converged"],"OLS passes":_opass,"NUTS diagnostics pass":_healthy,
                    "NUTS point error passes":_npass,"Classification":_classification,
                    "Posterior mass inside truth tolerance":_gate["posterior_mass_within_all_channel_tolerances"],
                    "Largest nonzero-channel 95% radius / tolerance":_gate["max_nonnull_radius95_over_allowance"],
                    "NUTS failing channels":_gate["failing_channels"]})
    nuts_ols_comparison_summary=[]
    for _mode in ("Estimated OLS","Constrained estimated OLS","True-transform OLS"):
        for _relative in (0.05,0.1):
            for _absolute in (0.005,0.01):
                _rows=[r for r in nuts_ols_comparison_rows if r["OLS method"]==_mode and r["Relative tolerance"]==_relative and r["Outcome tolerance"]==_absolute]
                nuts_ols_comparison_summary.append({"OLS method":_mode,"Relative tolerance":_relative,"Outcome tolerance":_absolute,"Worlds":len(_rows),
                    **{label:sum(r["Classification"]==label for r in _rows) for label in ("Both recover","NUTS only","OLS only","Neither recovers","NUTS unresolved")}})
    write_study_json(StudyPath("notebooks/bayesian_mmm_draw_diag/ols_recovery_comparison.json"),{"summary":nuts_ols_comparison_summary,"worlds":nuts_ols_comparison_rows})
    mo.vstack([mo.md("### OLS versus NUTS: recovery and uncertainty across 50 datasets\n\nDiagnostic failures are **unresolved**, not evidence that the task is intrinsically hard. NUTS-only recovery shows a limitation of this OLS procedure/assumptions, not a sampler-only advantage: NUTS also has informative priors and known reference anchors. Neither recovering, despite healthy NUTS diagnostics, is evidence of difficulty under this model and tolerance—not proof of non-identifiability.\n\nThe 95% radius is the posterior 95th percentile of each channel trajectory's L2 distance from its posterior mean, divided by its allowed recovery error. Large values indicate uncertainty wider than the required precision. Posterior mass inside the truth tolerance is a finite-draw descriptive estimate, not a calibrated success probability for new datasets."),
        mo.ui.table(nuts_ols_comparison_summary,selection=None),
        mo.ui.table([r for r in nuts_ols_comparison_rows if r["OLS method"]=="Estimated OLS" and r["Relative tolerance"]==0.1 and r["Outcome tolerance"]==0.01],selection=None)])
    return nuts_ols_comparison_rows, nuts_ols_comparison_summary


@app.cell
def nuts_full_study_conclusions(
    StudyPath,
    bayesian_reports,
    go,
    mo,
    np,
    nuts_ols_comparison_rows,
    nuts_ols_comparison_summary,
    qualitative,
    write_study_json,
):
    nuts50_focus_rows=[r for r in nuts_ols_comparison_rows if r["OLS method"]=="Estimated OLS" and r["Relative tolerance"]==0.1 and r["Outcome tolerance"]==0.01]
    nuts50_focus_summary=next(r for r in nuts_ols_comparison_summary if r["OLS method"]=="Estimated OLS" and r["Relative tolerance"]==0.1 and r["Outcome tolerance"]==0.01)
    nuts50_neither=[r for r in nuts50_focus_rows if r["Classification"]=="Neither recovers"]
    nuts50_unresolved=[{"World":r["world_id"]+1,"Divergences":r["divergences"],"Max Rhat":r["max_rhat"],
        "Min bulk ESS":r["min_bulk_ess"],"Min tail ESS":r["min_tail_ess"],"Max-depth terminations":r["maxdepth_hits"]}
        for r in bayesian_reports if not r["diagnostics_pass"]]
    nuts50_uncertainty={"neither_count":len(nuts50_neither),
        "largest_channel_radius_ratio_quantiles":np.quantile([r["Largest nonzero-channel 95% radius / tolerance"] for r in nuts50_neither],[0,0.5,1]).tolist(),
        "median_joint_truth_tolerance_mass":float(np.median([r["Posterior mass inside truth tolerance"] for r in nuts50_neither]))}
    nuts50_uncertainty_figure=go.Figure()
    for _i,_classification in enumerate(("Both recover","NUTS only","OLS only","Neither recovers","NUTS unresolved")):
        _rows=[r for r in nuts50_focus_rows if r["Classification"]==_classification]
        nuts50_uncertainty_figure.add_trace(go.Scatter(x=[r["World"] for r in _rows],y=[r["Largest nonzero-channel 95% radius / tolerance"] for r in _rows],
            mode="markers",name=_classification,marker={"size":9,"color":qualitative.Plotly[_i]},
            text=[f"Posterior mass inside truth tolerance: {r['Posterior mass inside truth tolerance']:.3f}" for r in _rows]))
    nuts50_uncertainty_figure.add_hline(y=1,line_dash="dash",annotation_text="Radius equals recovery tolerance")
    nuts50_uncertainty_figure.update_layout(height=500,template="plotly_white",title="Posterior precision versus recovery tolerance (10% / 1%)",
        xaxis_title="Study world",yaxis_title="Largest nonzero-channel 95% radius / tolerance",yaxis_type="log",
        legend={"orientation":"h","y":-0.18,"yanchor":"top"})
    _nuts50_path=StudyPath("notebooks/bayesian_mmm_draw_diag")
    write_study_json(_nuts50_path/"final_summary.json",{"comparison":nuts50_focus_summary,"uncertainty":nuts50_uncertainty,"unresolved":nuts50_unresolved,
        "sampling_minutes":sum(r["sampling_seconds"] for r in bayesian_reports)/60})
    _nuts50_text=f"""# 50-world oracle-graph NUTS / OLS comparison

    All 50 current datasets fitted with Nutpie draw_diag, 4 chains, 1000 tune + 1000 retained draws, target_accept .95 and maxdepth 10. Original five posterior files were reused unchanged. All fits know the nonzero effect identities; Bayesian inference also uses generator reference anchors and matching continuous priors. No new datasets or approximation fits were generated by this expansion.

    ## 10% relative / 1% outcome-scaled per-channel recovery

    {nuts50_focus_summary}

    Among the 46 fits passing the full NUTS diagnostic screen, 33 fail recovery with both methods. Their least precise nonzero channel has a 95% posterior radius about 3.21–8.20 times its recovery tolerance (median 5.04). This supports substantial attribution uncertainty under the current model/data/tolerances, not merely a failure of OLS optimization. It is not a proof of fundamental non-identifiability. The three NUTS-only recoveries also benefit from Bayesian prior/anchor information; this is not a sampler-only comparison. Point recovery and posterior precision are different properties.

    ## Unresolved fits (not counted as evidence of intrinsic task difficulty)

    - World 12: 803 divergences; 802 in chain 4, whose intercept/noise region disagrees with the other chains. Rhat about 1.55, bulk ESS about 7. Severe exploration failure; preserved, not repaired or excluded silently.
    - World 39: one divergence (large integration energy error), despite otherwise good Rhat/ESS. Fails the prespecified zero-divergence screen.
    - World 40: five maximum-depth terminations; otherwise good Rhat/ESS. Fails the prespecified depth screen.
    - World 50: Rhat about 1.021 and bulk ESS about 201, especially MM scales; insufficient mixing/precision at this budget.

    Total recorded sampling time over all 50 worlds: {sum(r['sampling_seconds'] for r in bayesian_reports)/60:.1f} minutes (includes the previously completed five).

    Full per-world results and all four threshold combinations: `ols_recovery_comparison.json`. Raw posteriors, warmup, diagnostic CSVs and uncertainty arrays are retained. Pathfinder and ADVI remain limited to the first five worlds.
    """
    (_nuts50_path/"README.md").write_text(_nuts50_text)
    mo.vstack([mo.md("### What does the full study indicate?\n\nMost failures persist with well-sampled NUTS and broad posterior uncertainty. Some OLS failures are rescued, but NUTS is not a universal point-recovery solution. The four unresolved fits must be investigated separately."),
        mo.ui.plotly(nuts50_uncertainty_figure),mo.ui.table(nuts50_unresolved,selection=None)])
    return


@app.cell(hide_code=True)
def tv_baseline_preparation(
    SlotLayout,
    StudyPath,
    bayesian_world_data,
    constant_baseline_cfg,
    mo,
    np,
    pg,
    replace,
    saved_study_records,
    showcase_cfg,
    study_hashlib,
    study_json,
    write_study_json,
):
    from pymc_generator.random_walk import _walk_basis as tv_walk_basis, walk_width_index as tv_width_index, symbolic_random_walk as tv_symbolic_walk
    from pymc_generator.slots import TRAJECTORY_COMPONENTS as tv_component_names
    from scipy.special import betainc as tv_beta_cdf, logsumexp as tv_logsumexp

    tv_baseline_cfg=replace(constant_baseline_cfg,rw_baseline_std_range=tuple(showcase_cfg.rw_baseline_std_range))
    assert tv_baseline_cfg.baseline_floor is None and tv_baseline_cfg.confounding_strength_range in (None,(0.0,0.0))
    assert tv_baseline_cfg.rw_baseline_std_range==(0.0,0.093)
    tv_full_steps=104+tv_baseline_cfg.carryover_burn_in
    tv_widths=np.arange(1,min(tv_full_steps,tv_baseline_cfg.rw_smoothness_max_weeks)+1)
    _edges=np.r_[0,(tv_widths[:-1]+.5)/tv_baseline_cfg.rw_smoothness_max_weeks,1]
    tv_width_prior=np.diff(tv_beta_cdf(tv_baseline_cfg.rw_smoothness_alpha,tv_baseline_cfg.rw_smoothness_beta,_edges))
    assert np.all(tv_width_prior>0) and np.isclose(tv_width_prior.sum(),1)
    tv_bases=np.stack([tv_walk_basis(tv_full_steps,int(w)) for w in tv_widths])
    _tv_cov=tv_bases[:,15:,:]@tv_bases[:,15:,:].transpose(0,2,1)
    _tv_eigenvalues,tv_eigenvectors=np.linalg.eigh(_tv_cov)
    assert _tv_eigenvalues.min()>-1e-10
    tv_eigenvalues=np.maximum(_tv_eigenvalues,0)
    np.testing.assert_allclose((tv_eigenvectors*tv_eigenvalues[:,None,:])@tv_eigenvectors.transpose(0,2,1),_tv_cov,atol=1e-10,rtol=1e-10)
    tv_projection=tv_eigenvectors.transpose(0,2,1).reshape(-1,97)


    def prepare_matched_tv_worlds():
        directory=StudyPath("notebooks/time_varying_baseline");directory.mkdir(exist_ok=True)
        source=StudyPath("notebooks/ols_study_50_seed10029")
        manifest={"baseline_std_range":list(tv_baseline_cfg.rw_baseline_std_range),"smoothness_beta_prior":[tv_baseline_cfg.rw_smoothness_alpha,tv_baseline_cfg.rw_smoothness_beta],
            "full_horizon":tv_full_steps,"reported_window":[8,112],"fit_window":[15,112],"max_smoothing_width":tv_baseline_cfg.rw_smoothness_max_weeks,
            "baseline_floor":None,"confounding_strength":0,"matched":"Preserve inputs, graph, parameters, baseline level and observation-noise realization; independently draw baseline scale, smoothness and innovations from generator prior.",
            "seeds":list(range(91001,91051)),"source_sha256":{p.name:study_hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob("batch_*.npz"))}}
        if (directory/"data_manifest.json").exists():
            assert study_json.loads((directory/"data_manifest.json").read_text())==manifest
        else:write_study_json(directory/"data_manifest.json",manifest)
        records=[];patterns=[]
        for record in saved_study_records:
            w=record["world_id"];local=record["local_world"]
            corpus=pg.load_corpus(source/f"batch_{record['batch']:02d}.npz")
            assert float(corpus["confounding_strength"][local])==0
            rng=np.random.default_rng(91001+w)
            smoothness=float(rng.beta(tv_baseline_cfg.rw_smoothness_alpha,tv_baseline_cfg.rw_smoothness_beta))
            width=int(tv_width_index([smoothness],tv_full_steps,rw_smoothness_max_weeks=tv_baseline_cfg.rw_smoothness_max_weeks)[0])+1
            relative=float(rng.uniform(*tv_baseline_cfg.rw_baseline_std_range))
            graph=SlotLayout(15,10,1).unpack(corpus["g"][local])
            amplitude=float(np.linalg.norm(corpus["beta"][local].astype(float)*graph["cy"]))
            innovations=rng.normal(size=tv_full_steps)
            baseline_full=float(corpus["baseline_intrinsic"][local,0])+relative*amplitude*(tv_bases[width-1]@innovations)
            baseline=baseline_full[8:]
            observed=corpus["outcome_raw"][local].astype(float)+baseline-corpus["baseline_intrinsic"][local].astype(float)
            path=directory/f"world_{w:02d}_data.npz"
            if path.exists():
                with np.load(path) as previous:np.testing.assert_array_equal(previous["baseline"],baseline)
            else:np.savez_compressed(path,baseline=baseline,observed=observed,baseline_full=baseline_full,innovations=innovations)
            new={**record,"baseline_seed":91001+w,"true_baseline_relative_sd":relative,"true_baseline_sd":relative*amplitude,"true_smoothness":smoothness,"true_width":width}
            records.append(new)
            for kind,mask,flags,activity,input_values,effects in (("media",corpus["treatment_active_mask"][local],corpus["treatment_components"][local],corpus["treatment_activity"][local],corpus["treatment_raw"][local],graph["cy"]),
                    ("control",corpus["covariate_active_mask"][local],corpus["covariate_components"][local],corpus["covariate_activity"][local],corpus["covariates"][local],graph["zy"])):
                for slot in np.flatnonzero(mask):
                    item={"world_id":w,"kind":kind,"channel":("C" if kind=="media" else "Z")+str(slot+1),"nonzero_effect":bool(effects[slot]),
                        "all_off_fitted_window":bool(np.all(activity[7:,slot]==0)),"all_zero_input_fitted_window":bool(np.all(input_values[7:,slot]==0))}
                    item.update({name:bool(flags[slot,j]) for j,name in enumerate(tv_component_names)})
                    if item.get("trend"):
                        change=corpus[f"trajectory_{'treatment' if kind=='media' else 'covariate'}_trend_change"][local,slot]
                        item["trend_direction"]="up" if change>0 else "down" if change<0 else "flat"
                    patterns.append(item)
        write_study_json(directory/"records.json",records);write_study_json(directory/"patterns.json",patterns)
        return records,patterns


    def tv_world_data(record):
        data,corpus=bayesian_world_data(record)
        with np.load(StudyPath("notebooks/time_varying_baseline")/f"world_{record['world_id']:02d}_data.npz") as a:
            outcome=a["observed"][7:].copy()
        data["outcome_scale"]=np.array(max(float(outcome.std()),1e-8))
        data["outcome_data"]=outcome/data["outcome_scale"]
        return data,corpus

    tv_records,tv_patterns=prepare_matched_tv_worlds()
    mo.md("Matched time-varying-baseline worlds prepared. Original inputs, graph, effects, baseline level, and observation noise are unchanged. Baseline amplitude and smoothing width remain unknown to inference; only their generator priors are supplied.")

    return (
        tv_baseline_cfg,
        tv_bases,
        tv_eigenvalues,
        tv_eigenvectors,
        tv_logsumexp,
        tv_patterns,
        tv_projection,
        tv_records,
        tv_symbolic_walk,
        tv_width_index,
        tv_width_prior,
        tv_widths,
        tv_world_data,
    )


@app.cell(hide_code=True)
def tv_baseline_model(
    bayes_pm,
    mechanisms,
    mo,
    np,
    pt,
    tv_baseline_cfg,
    tv_eigenvalues,
    tv_projection,
    tv_records,
    tv_width_prior,
    tv_widths,
    tv_world_data,
):
    def build_tv_baseline_model(data):
        from pytensor.graph.rewriting.utils import rewrite_graph
        cfg=tv_baseline_cfg
        coords = {"week":np.arange(104),"fit_week":np.arange(7,104),"media":np.arange(15),"control":np.arange(10)}
        assert cfg.mm_scale_prior=="log_uniform" and cfg.treatment_reference_multiplier==1.0
        coords["baseline_width_option"]=tv_widths
        with bayes_pm.Model(coords=coords) as model:
            x=bayes_pm.Data("media_data",data["media_data"],dims=("week","media"))
            z=bayes_pm.Data("control_data",data["control_data"],dims=("week","control"))
            media_scale=bayes_pm.Data("media_scale",data["media_scale"],dims="media")
            control_scale=bayes_pm.Data("control_scale",data["control_scale"],dims="control")
            media_mask=bayes_pm.Data("media_mask",data["media_mask"],dims="media")
            control_mask=bayes_pm.Data("control_mask",data["control_mask"],dims="control")
            media_effect_mask=bayes_pm.Data("media_effect_mask",data["media_effect_mask"],dims="media")
            control_effect_mask=bayes_pm.Data("control_effect_mask",data["control_effect_mask"],dims="control")
            anchor=bayes_pm.Data("saturation_anchor",data["saturation_anchor"],dims="media")
            y_scale=bayes_pm.Data("outcome_scale",data["outcome_scale"])
            y=bayes_pm.Data("outcome_data",data["outcome_data"],dims="fit_week")
            intercept=bayes_pm.Uniform("intercept",*cfg.rw_baseline_mean_range)
            alpha=bayes_pm.Uniform("alpha",*cfg.carryover_alpha_range,dims="media")
            k_lo,k_hi=cfg.saturation_prior_ranges["michaelis_menten"]["kappa_mult"]
            log_kappa=bayes_pm.Uniform("log_kappa",np.log(k_lo),np.log(k_hi),dims="media")
            kappa=bayes_pm.Deterministic("mm_kappa_mult",pt.clip(pt.exp(log_kappa),k_lo,k_hi),dims="media")
            half=bayes_pm.Deterministic("half_saturation",kappa*anchor,dims="media")
            half_ratio=bayes_pm.Deterministic("half_ratio",half/media_scale,dims="media")
            amplitude=bayes_pm.Uniform("amplitude",*cfg.treatment_reference_contribution_range,dims="media")
            # Effective coefficients include the known graph, so all null contributions are exactly zero.
            beta=bayes_pm.Deterministic("beta",amplitude*(1+kappa)*media_effect_mask,dims="media")
            control_reference=bayes_pm.Uniform("control_reference_contribution",*cfg.covariate_reference_contribution_range,dims="control")
            rho=bayes_pm.Deterministic("rho",control_reference/cfg.covariate_reference_scale*control_effect_mask,dims="control")
            control_amplitude=bayes_pm.Deterministic("control_amplitude",rho*control_scale,dims="control")
            responses=pt.stack([
                mechanisms.stable_michaelis_menten_kappa_relative(
                    mechanisms.apply_geometric_carryover(x[:,channel:channel+1],alpha[channel],8)[:,0],
                    1.0,kappa_mult=half_ratio[channel]) for channel in range(15)
            ],axis=1)[7:]
            responses=rewrite_graph(responses,include=("lower_xtensor",),clone=False)
            media_effect=responses*beta*media_mask
            control_effect=z[7:]*control_amplitude*control_mask
            bayes_pm.Deterministic("mean_media_contribution",media_effect.mean(axis=0),dims="media")
            bayes_pm.Deterministic("mean_control_contribution",control_effect.mean(axis=0),dims="control")
            mu=bayes_pm.Deterministic("mu",intercept+media_effect.sum(axis=1)+control_effect.sum(axis=1),dims="fit_week")
            sigma_relative=bayes_pm.Uniform("sigma_relative",*cfg.rw_outcome_std_range)
            sigma=bayes_pm.Deterministic("sigma",sigma_relative*pt.sqrt(pt.sum(beta**2)))
            # Exact finite Gaussian mixture: integrate baseline innovations and smoothing width.
            baseline_relative=bayes_pm.Uniform("baseline_sd_relative",*cfg.rw_baseline_std_range)
            baseline_sd=bayes_pm.Deterministic("baseline_sd",baseline_relative*pt.sqrt(pt.sum(beta**2)))
            residual=y*y_scale-mu
            projected=pt.dot(pt.as_tensor_variable(tv_projection),residual).reshape((len(tv_widths),97))
            variance=baseline_sd**2*pt.as_tensor_variable(tv_eigenvalues)+sigma**2
            width_logp=-0.5*pt.sum(projected**2/variance+pt.log(variance)+np.log(2*np.pi),axis=1)
            logits=pt.log(pt.as_tensor_variable(tv_width_prior))+width_logp
            marginal_logp=pt.logsumexp(logits)
            bayes_pm.Deterministic("baseline_width_probabilities",pt.exp(logits-marginal_logp),dims="baseline_width_option")
            bayes_pm.Potential("baseline_marginal_likelihood",marginal_logp)
        return model



    tv_initial_data,_tv_initial_corpus=tv_world_data(tv_records[0])
    tv_model=build_tv_baseline_model(tv_initial_data)
    tv_initial_logp=tv_model.compile_logp()(tv_model.initial_point())
    tv_initial_gradient=tv_model.compile_dlogp()(tv_model.initial_point())
    assert np.isfinite(tv_initial_logp) and np.isfinite(tv_initial_gradient).all()
    mo.md(f"Exact marginalized baseline model: finite initial log density {tv_initial_logp:.2f}; all gradients finite. No true baseline amplitude or smoothing width is supplied to inference.")

    return build_tv_baseline_model, tv_initial_gradient, tv_model


@app.cell(hide_code=True)
def tv_baseline_validation_compile(
    StudyPath,
    bayes_nutpie,
    bayes_pd,
    bayes_pm,
    mo,
    np,
    perf_counter,
    pg,
    pt,
    transform_media_columns,
    tv_bases,
    tv_initial_gradient,
    tv_logsumexp,
    tv_model,
    tv_records,
    tv_symbolic_walk,
    tv_width_index,
    tv_width_prior,
    tv_world_data,
    write_study_json,
):
    from scipy.stats import multivariate_normal as tv_mvn
    from threadpoolctl import threadpool_limits as tv_thread_limits

    _tv_data,_=tv_world_data(tv_records[0])
    _beta=np.full(15,2.5)*_tv_data["media_effect_mask"]
    _response=transform_media_columns(_tv_data["media_data"],np.full(15,.5),_tv_data["saturation_anchor"]/_tv_data["media_scale"])[7:]
    _mu=5.5+(_response*_beta).sum(axis=1)
    _sigma=.019*np.linalg.norm(_beta);_tau=.0465*np.linalg.norm(_beta)
    _cov=tv_bases[:,15:,:]@tv_bases[:,15:,:].transpose(0,2,1)
    with tv_thread_limits(limits=1):
        _logpdf=np.array([tv_mvn.logpdf(_tv_data["outcome_data"]*_tv_data["outcome_scale"],mean=_mu,cov=_tau**2*c+_sigma**2*np.eye(97)) for c in _cov])
    _tv_numpy_logp=float(tv_logsumexp(np.log(tv_width_prior)+_logpdf))
    _tv_symbolic_logp=float(tv_model.compile_logp(vars=tv_model.potentials,jacobian=False)(tv_model.initial_point()))
    np.testing.assert_allclose(_tv_numpy_logp,_tv_symbolic_logp,atol=1e-8,rtol=1e-10)
    _point=tv_model.initial_point();_logp=tv_model.compile_logp();_numeric=[]
    for _variable in tv_model.value_vars:
        _name=_variable.name
        for _i in range(np.asarray(_point[_name]).size):
            _plus={k:np.array(v,copy=True) for k,v in _point.items()};_minus={k:np.array(v,copy=True) for k,v in _point.items()}
            _plus[_name].flat[_i]+=1e-5;_minus[_name].flat[_i]-=1e-5
            _numeric.append((_logp(_plus)-_logp(_minus))/(2e-5))
    np.testing.assert_allclose(tv_initial_gradient,_numeric,atol=2e-5,rtol=2e-5)
    _rng=np.random.default_rng(91700)
    _width_draws=tv_width_index(_rng.beta(2,2,size=100000),112,rw_smoothness_max_weeks=26)
    _width_frequency=np.bincount(_width_draws,minlength=26)/100000
    assert np.max(np.abs(_width_frequency-tv_width_prior))<.004
    for _w in (0,49):
        _record=tv_records[_w]
        with np.load(StudyPath("notebooks/time_varying_baseline")/f"world_{_w:02d}_data.npz") as _a:
            _original=pg.load_corpus(StudyPath("notebooks/ols_study_50_seed10029")/f"batch_{_record['batch']:02d}.npz")
            _mean=float(_original["baseline_intrinsic"][_record["local_world"],0])
            _symbolic=tv_symbolic_walk(112,_mean,_record["true_baseline_sd"],_record["true_smoothness"],False,rw_smoothness_max_weeks=26,eps=pt.as_tensor_variable(_a["innovations"])).eval()
            np.testing.assert_allclose(_symbolic,_a["baseline_full"],rtol=1e-10,atol=1e-10)
            _nonbaseline=_original["outcome_raw"][_record["local_world"]].astype(float)-_original["baseline_intrinsic"][_record["local_world"]].astype(float)
            np.testing.assert_allclose(_a["observed"]-_a["baseline"],_nonbaseline,atol=1e-12)
    # Prior simulation explicitly draws the marginalized width and walk, not posterior width weights.
    with tv_model:
        _prior_mu,_prior_sigma,_prior_tau=bayes_pm.draw([tv_model["mu"],tv_model["sigma"],tv_model["baseline_sd"]],draws=100,random_seed=91701)
    _prior_width=_rng.choice(26,size=100,p=tv_width_prior)
    _prior_baseline=np.stack([_prior_tau[i]*(tv_bases[k,15:]@_rng.normal(size=112)) for i,k in enumerate(_prior_width)])
    _prior_y=_prior_mu+_prior_baseline+_rng.normal(size=_prior_mu.shape)*_prior_sigma[:,None]
    assert np.isfinite(_prior_y).all()
    tv_model_validation={"likelihood_numpy":_tv_numpy_logp,"likelihood_symbolic":_tv_symbolic_logp,"max_gradient_difference":float(np.max(np.abs(tv_initial_gradient-_numeric))),
        "width_prior_max_frequency_gap":float(np.max(np.abs(_width_frequency-tv_width_prior))),"baseline_symbolic_parity_worlds":[1,50],
        "prior_predictive_quantiles_05_50_95":np.quantile(_prior_y,[.05,.5,.95]).tolist(),"prior_predictive_negative_fraction":float(np.mean(_prior_y<0))}
    write_study_json(StudyPath("notebooks/time_varying_baseline/model_validation.json"),tv_model_validation)
    _tv_compile_started=perf_counter()
    with bayes_pd.option_context("future.infer_string",False),tv_thread_limits(limits=1):
        tv_compiled=bayes_nutpie.compile_pymc_model(tv_model,backend="numba")
    tv_compile_seconds=perf_counter()-_tv_compile_started
    mo.md(f"Time-varying baseline model validated against independent dense-Gaussian likelihoods, finite differences and the library's symbolic walk. Nutpie compilation: {tv_compile_seconds:.1f}s.")

    return tv_compile_seconds, tv_compiled, tv_thread_limits


@app.cell(hide_code=True)
def tv_baseline_inference_functions(
    StudyPath,
    bayes_az,
    bayes_nutpie,
    bayes_xr,
    build_tv_baseline_model,
    channel_gate_metrics,
    np,
    perf_counter,
    posterior_component_draws,
    study_inspect,
    study_json,
    tv_compile_seconds,
    tv_compiled,
    tv_eigenvalues,
    tv_eigenvectors,
    tv_records,
    tv_thread_limits,
    tv_widths,
    tv_world_data,
    write_study_json,
):
    def reconstruct_tv_posterior(idata,data,record,seed):
        components,static_mu=posterior_component_draws(idata,data,record)
        post=idata["posterior"];shape=post["intercept"].shape
        sigma=post["sigma"].values.reshape(-1);tau=post["baseline_sd"].values.reshape(-1)
        probabilities=post["baseline_width_probabilities"].transpose("chain","draw","baseline_width_option").values.reshape(-1,26)
        np.testing.assert_allclose(probabilities.sum(axis=1),1,atol=1e-10)
        rng=np.random.default_rng(seed)
        widths=np.minimum((rng.random(len(sigma))[:,None]>probabilities.cumsum(axis=1)).sum(axis=1),25)
        residual=data["outcome_data"]*data["outcome_scale"]-static_mu
        deviation=np.empty_like(residual)
        for width in np.unique(widths):
            ix=np.flatnonzero(widths==width)
            u=tv_eigenvectors[width];eigen=tv_eigenvalues[width]
            latent_variance=tau[ix,None]**2*eigen
            variance=latent_variance+sigma[ix,None]**2
            gain=latent_variance/variance
            conditional_mean=(residual[ix]@u)*gain
            conditional_sd=np.sqrt(latent_variance*sigma[ix,None]**2/variance)
            deviation[ix]=(conditional_mean+conditional_sd*rng.normal(size=conditional_mean.shape))@u.T
        components[:,:,0]+=deviation
        mu=static_mu+deviation
        np.testing.assert_allclose(components.sum(axis=2),mu,rtol=1e-10,atol=1e-10)
        # These are conditional posterior draws, not independently added prior walks.
        idata["posterior"]["baseline_path"]=bayes_xr.DataArray(components[:,:,0].reshape(*shape,97),dims=("chain","draw","fit_week"))
        idata["posterior"]["baseline_expected_width"]=bayes_xr.DataArray((probabilities@tv_widths).reshape(shape),dims=("chain","draw"))
        return components,mu,widths.reshape(shape)


    def summarize_tv_fit(idata,data,corpus,record,directory,label,*,mcmc=True):
        world=record["world_id"]
        components,mu,widths=reconstruct_tv_posterior(idata,data,record,93001+world)
        with np.load(StudyPath("notebooks/ols_study_50_seed10029")/f"world_{world:02d}_fits.npz") as original:
            truth=original["truth"].copy()
        with np.load(StudyPath("notebooks/time_varying_baseline")/f"world_{world:02d}_data.npz") as generated:
            truth[:,0]=generated["baseline"][7:]
            observed=generated["observed"][7:].copy()
        point=components.mean(axis=0);low,high=np.quantile(components,[.025,.975],axis=0)
        healthy=None;diagnostics={}
        if mcmc:
            media=np.flatnonzero(data["media_effect_mask"]);control=np.flatnonzero(data["control_effect_mask"])
            variables=["intercept","sigma","sigma_relative","alpha","mm_kappa_mult","amplitude","mean_media_contribution","baseline_sd_relative","baseline_sd","baseline_expected_width","baseline_path"]
            if len(control):variables.append("control_reference_contribution")
            summary=bayes_az.summary(idata,var_names=variables,coords={"media":media,**({"control":control} if len(control) else {})},ci_prob=.95,ci_kind="eti",round_to="none")
            summary.to_csv(directory/f"world_{world:02d}_{label}_diagnostics.csv")
            stats=idata["sample_stats"];energy=stats["energy"].values
            bfmi=np.mean(np.diff(energy,axis=1)**2,axis=1)/np.var(energy,axis=1,ddof=1)
            diagnostics={"divergences":int(stats["diverging"].sum()),"maxdepth_hits":int(stats["maxdepth_reached"].sum()),"max_rhat":float(summary["r_hat"].max()),"min_bulk_ess":float(summary["ess_bulk"].min()),"min_tail_ess":float(summary["ess_tail"].min()),"min_bfmi":float(bfmi.min())}
            finite=np.isfinite(summary[["r_hat","ess_bulk","ess_tail"]].to_numpy()).all() and np.isfinite(bfmi).all()
            healthy=bool(finite and diagnostics["divergences"]==0 and diagnostics["maxdepth_hits"]==0 and diagnostics["max_rhat"]<1.01 and diagnostics["min_bulk_ess"]>=400 and diagnostics["min_tail_ess"]>=400 and diagnostics["min_bfmi"]>=.3)
        media_indices=[i for i,n in enumerate(record["terms"]) if n.startswith("C")]
        media_draws=components[:,:,media_indices].sum(axis=2)
        media_truth=truth[:,media_indices].sum(axis=1)
        noiseless=truth.sum(axis=1)
        gates=[]
        for relative in (.05,.1):
            for absolute in (.005,.01):
                _,_,allowed,passed,ratios=channel_gate_metrics(truth[:,1:],point[:,1:],np.linalg.norm(noiseless),relative,absolute)
                metrics={}
                for name,draws,actual in (("baseline",components[:,:,0],truth[:,0]),("baseline_shape",components[:,:,0]-components[:,:,0].mean(axis=1,keepdims=True),truth[:,0]-truth[:,0].mean()),("total_media",media_draws,media_truth)):
                    mean=draws.mean(axis=0);allowance=max(relative*np.linalg.norm(actual),absolute*np.linalg.norm(noiseless))
                    metrics[name+"_error_ratio"]=float(np.linalg.norm(mean-actual)/allowance)
                    metrics[name+"_point_pass"]=bool(metrics[name+"_error_ratio"]<=1)
                    metrics[name+"_radius95_ratio"]=float(np.quantile(np.linalg.norm(draws-mean,axis=1),.95)/allowance)
                gates.append({"relative":relative,"absolute":absolute,"individual_point_pass":bool(passed.all()),"media_point_pass":bool(passed[:record["media"]].all()),
                    "failing_channels":[n for n,ok in zip(record["terms"][1:],passed) if not ok],**metrics})
        sigma=idata["posterior"]["sigma"].values.reshape(-1)
        predictive=mu+np.random.default_rng(94001+world).normal(size=mu.shape)*sigma[:,None]
        pl,pu=np.quantile(predictive,[.025,.975],axis=0)
        expected_low,expected_high=np.quantile(mu,[.025,.975],axis=0)
        ml,mu_upper=np.quantile(media_draws,[.025,.975],axis=0)
        np.savez_compressed(directory/f"world_{world:02d}_{label}_recovery.npz",truth=truth,mean=point,lower95=low,upper95=high,observed=observed,noiseless=noiseless,
            predictive_lower95=pl,predictive_upper95=pu,expected_lower95=expected_low,expected_upper95=expected_high,total_media_truth=media_truth,total_media_mean=media_draws.mean(axis=0),total_media_lower95=ml,total_media_upper95=mu_upper)
        bayes_xr.Dataset({"baseline_path":(("chain","draw","fit_week"),idata["posterior"]["baseline_path"].values),"width":(("chain","draw"),widths+1)},coords={"fit_week":np.arange(7,104)}).to_netcdf(directory/f"world_{world:02d}_{label}_baseline_draws.nc",engine="h5netcdf")
        return {"world_id":world,"method":label,"diagnostics_pass":healthy,**diagnostics,"gates":gates,
            "baseline_pointwise_95pct_coverage":float(np.mean((truth[:,0]>=low[:,0])&(truth[:,0]<=high[:,0]))),
            "observed_predictive_95pct_coverage":float(np.mean((observed>=pl)&(observed<=pu))),"terms":["Baseline"]+record["terms"][1:]}


    def run_tv_baseline_nuts(limit=5):
        directory=StudyPath("notebooks/time_varying_baseline/nuts");directory.mkdir(exist_ok=True)
        manifest={"data_manifest":study_json.loads(StudyPath("notebooks/time_varying_baseline/data_manifest.json").read_text()),
            "model_source":study_inspect.getsource(build_tv_baseline_model),"sampler":"nutpie","adaptation":"draw_diag","chains":4,"tune":1000,"draws":1000,"target_accept":.95,"maxdepth":10,"seeds":list(range(92001,92051)),
            "baseline_inference":"Exact marginalization of Gaussian walk and discrete Beta-induced width prior; conditional posterior path reconstruction."}
        if (directory/"manifest.json").exists():assert study_json.loads((directory/"manifest.json").read_text())==manifest
        else:write_study_json(directory/"manifest.json",manifest)
        reports=[]
        for record in tv_records[:limit]:
            w=record["world_id"];data,corpus=tv_world_data(record)
            path=directory/f"world_{w:02d}_nuts_posterior.nc";timing_path=directory/f"world_{w:02d}_timing.json"
            cached=path.exists()
            if cached:
                idata=bayes_az.from_netcdf(path,engine="h5netcdf");timing=study_json.loads(timing_path.read_text())
            else:
                print(f"Starting time-varying baseline NUTS world {w+1}/{limit}",flush=True)
                started=perf_counter()
                with tv_thread_limits(limits=1):
                    idata=bayes_nutpie.sample(tv_compiled.with_data(**data),chains=4,cores=4,tune=1000,draws=1000,seed=92001+w,adaptation="draw_diag",target_accept=.95,maxdepth=10,save_warmup=True,progress_bar=False)
                timing={"sampling_seconds":perf_counter()-started,"shared_compilation_seconds":tv_compile_seconds}
                idata["observed_data"]=bayes_xr.Dataset({"outcome_raw":("fit_week",data["outcome_data"]*data["outcome_scale"])},coords={"fit_week":np.arange(7,104)})
                idata["constant_data"]=bayes_xr.Dataset({"saturation_anchor":("media",data["saturation_anchor"]),"media_effect_mask":("media",data["media_effect_mask"]),"control_effect_mask":("control",data["control_effect_mask"])},coords={"media":np.arange(15),"control":np.arange(10)})
                idata.attrs.update(world_id=w,adaptation="draw_diag",baseline_model="generator smoothed centered random walk; exact width/path marginalization")
                write_study_json(timing_path,timing);idata.to_netcdf(path,engine="h5netcdf")
            report=summarize_tv_fit(idata,data,corpus,record,directory,"nuts");report.update(timing)
            write_study_json(directory/f"world_{w:02d}_nuts_report.json",report);reports.append(report)
            write_study_json(directory/"reports.json",reports)
            write_study_json(directory/"progress.json",{"completed":len(reports),"requested":limit,"planned":50,"diagnostics_pass":sum(r["diagnostics_pass"] for r in reports)})
            print(f"TV world {w+1} ({'cached' if cached else 'new'}): {timing['sampling_seconds']:.1f}s, valid={report['diagnostics_pass']}, divergences={report['divergences']}, Rhat={report['max_rhat']:.4f}, ESS={report['min_bulk_ess']:.0f}",flush=True)
        return reports


    return run_tv_baseline_nuts, summarize_tv_fit


@app.cell(hide_code=True)
def tv_baseline_pilot(run_tv_baseline_nuts):
    tv_pilot_reports=run_tv_baseline_nuts(5)
    return


@app.cell(hide_code=True)
def tv_baseline_advi_functions(
    StudyPath,
    bayes_az,
    bayes_pm,
    build_tv_baseline_model,
    np,
    perf_counter,
    study_inspect,
    study_json,
    summarize_tv_fit,
    tv_records,
    tv_thread_limits,
    tv_world_data,
    write_study_json,
):
    def run_tv_baseline_advi():
        directory=StudyPath("notebooks/time_varying_baseline/advi");directory.mkdir(exist_ok=True)
        manifest={"model_source":study_inspect.getsource(build_tv_baseline_model),"data_manifest":study_json.loads(StudyPath("notebooks/time_varying_baseline/data_manifest.json").read_text()),
            "method":"fullrank_advi","worlds":list(range(5)),"replicates":2,"steps":20000,"draws":4000,"obj_n_mc":1,"optimizer":"default adagrad_window"}
        if (directory/"manifest.json").exists():assert study_json.loads((directory/"manifest.json").read_text())==manifest
        else:write_study_json(directory/"manifest.json",manifest)
        reports=[]
        for record in tv_records[:5]:
            w=record["world_id"];data,corpus=tv_world_data(record)
            model=build_tv_baseline_model(data)
            for replicate in range(2):
                label=f"fullrank_advi_{replicate+1}";seed=95001+10*w+replicate
                prefix=directory/f"world_{w:02d}_{label}"
                posterior_path=directory/f"world_{w:02d}_{label}_posterior.nc";metadata_path=directory/f"world_{w:02d}_{label}_metadata.json"
                if posterior_path.exists():
                    idata=bayes_az.from_netcdf(posterior_path,engine="h5netcdf");metadata=study_json.loads(metadata_path.read_text())
                else:
                    print(f"Starting TV {label}, world {w+1}/5",flush=True)
                    began=perf_counter()
                    with model,tv_thread_limits(limits=1):
                        inference=bayes_pm.FullRankADVI(random_seed=seed)
                        inference.fit(n=0,score=True,progressbar=False,obj_n_mc=1)
                        compile_seconds=perf_counter()-began
                        fit_start=perf_counter();inference.refine(5000,progressbar=False)
                        checkpoint_loss=float(np.mean(inference.hist[-1000:]))
                        inference.refine(15000,progressbar=False)
                        fit_seconds=perf_counter()-fit_start
                        np.savez_compressed(directory/f"world_{w:02d}_{label}_fit.npz",loss=inference.hist.copy(),**{f"parameter_{i}":p.get_value() for i,p in enumerate(inference.approx.params)})
                        idata=inference.approx.sample(draws=4000,random_seed=seed+1000)
                    metadata={"seed":seed,"compile_seconds":compile_seconds,"fit_seconds":fit_seconds,"total_seconds":perf_counter()-began,"iterations":len(inference.hist),
                        "finite_loss":bool(np.isfinite(inference.hist).all()),"checkpoint_5000_mean_loss":checkpoint_loss,"last_1000_mean_loss":float(np.mean(inference.hist[-1000:])),"previous_1000_mean_loss":float(np.mean(inference.hist[-2000:-1000]))}
                    write_study_json(metadata_path,metadata);idata.to_netcdf(posterior_path,engine="h5netcdf")
                report=summarize_tv_fit(idata,data,corpus,record,directory,label,mcmc=False);report.update(metadata)
                with np.load(StudyPath("notebooks/time_varying_baseline/nuts")/f"world_{w:02d}_nuts_recovery.npz") as n,np.load(directory/f"world_{w:02d}_{label}_recovery.npz") as a:
                    width=n["upper95"]-n["lower95"];valid=width>1e-12
                    report["median_interval_width_ratio_to_nuts"]=float(np.median(((a["upper95"]-a["lower95"])[valid])/width[valid]))
                    report["median_mean_gap_in_nuts_widths"]=float(np.median(np.abs(a["mean"]-n["mean"])[valid]/width[valid]))
                write_study_json(directory/f"world_{w:02d}_{label}_report.json",report);reports.append(report)
                write_study_json(directory/"reports.json",reports)
                print(f"TV {label} world {w+1}: {report['total_seconds']:.1f}s, interval ratio {report['median_interval_width_ratio_to_nuts']:.3f}",flush=True)
        return reports


    return (run_tv_baseline_advi,)


@app.cell(hide_code=True)
def tv_baseline_advi_pilot(run_tv_baseline_advi):
    tv_advi_reports=run_tv_baseline_advi()
    return (tv_advi_reports,)


@app.cell(hide_code=True)
def tv_baseline_full_study(run_tv_baseline_nuts):
    tv_full_reports=run_tv_baseline_nuts(50)
    return (tv_full_reports,)


@app.cell(hide_code=True)
def tv_baseline_study_summary(
    StudyPath,
    bayesian_reports,
    mo,
    np,
    total_media_primary,
    tv_advi_reports,
    tv_full_reports,
    tv_patterns,
    write_study_json,
):
    tv_primary_gates={r["world_id"]:next(g for g in r["gates"] if g["relative"]==.1 and g["absolute"]==.01) for r in tv_full_reports}
    _tv_common=[r["world_id"] for r in tv_full_reports if r["diagnostics_pass"] and bayesian_reports[r["world_id"]]["diagnostics_pass"]]
    tv_paired_summary=[]
    for _label,_key in (("All individual media","media_point_pass"),("All media + controls","individual_point_pass"),("Total media","total_media_point_pass")):
        _old=[];_new=[]
        for _w in _tv_common:
            _g=next(g for g in bayesian_reports[_w]["gates"] if g["relative"]==.1 and g["absolute"]==.01)
            _old.append(_g["media_pass"] if _key=="media_point_pass" else _g["error_gates_pass"] if _key=="individual_point_pass" else next(r for r in total_media_primary if r["World"]==_w+1 and r["Method"]=="NUTS")["Total media point pass"])
            _new.append(tv_primary_gates[_w][_key])
        tv_paired_summary.append({"Target":_label,"Matched valid worlds":len(_tv_common),"Constant baseline passes":sum(_old),"Varying baseline passes":sum(_new),
            "Both pass":sum(a and b for a,b in zip(_old,_new)),"Constant only":sum(a and not b for a,b in zip(_old,_new)),"Varying only":sum(b and not a for a,b in zip(_old,_new))})
    tv_overall_summary=[]
    for _label,_key in (("Individual media","media_point_pass"),("Media + controls","individual_point_pass"),("Total media","total_media_point_pass"),("Baseline path","baseline_point_pass"),("Centered baseline shape","baseline_shape_point_pass")):
        tv_overall_summary.append({"Target":_label,"Passes":sum(g[_key] for g in tv_primary_gates.values()),"Worlds":len(tv_full_reports)})
    tv_pattern_summary=[]
    for _label,_key,_direction in (("On/off flighting","flighting",None),("Seasonal","seasonal",None),("Ramping up","trend","up"),("Ramping down","trend","down"),("Delayed start","onset",None),("Early end","offset",None),("Level jump","level_jump",None),("All-off for entire fitted window","all_off_fitted_window",None)):
        _channels=[p for p in tv_patterns if p["kind"]=="media" and p["nonzero_effect"] and p.get(_key,False) and (_direction is None or p.get("trend_direction")==_direction)]
        _present={p["world_id"] for p in _channels};_absent=set(range(50))-_present
        _common_present=_present&set(_tv_common)
        tv_pattern_summary.append({"Pattern":_label,"Contributing media channels":len(_channels),"Worlds with pattern":len(_present),
            "Individual-media failures with pattern":sum(not tv_primary_gates[w]["media_point_pass"] for w in _present),"Worlds without pattern":len(_absent),
            "Individual-media failures without pattern":sum(not tv_primary_gates[w]["media_point_pass"] for w in _absent),"Total-media failures with pattern":sum(not tv_primary_gates[w]["total_media_point_pass"] for w in _present),
            "Common-valid worlds with pattern":len(_common_present),"Constant-baseline individual-media failures in those worlds":sum(not next(g for g in bayesian_reports[w]["gates"] if g["relative"]==.1 and g["absolute"]==.01)["media_pass"] for w in _common_present)})
    tv_realism_flags=[]
    for _w in range(50):
        with np.load(StudyPath("notebooks/time_varying_baseline")/f"world_{_w:02d}_data.npz") as _a:
            if _a["baseline"][7:].min()<0 or _a["observed"][7:].min()<0:
                tv_realism_flags.append({"World":_w+1,"Minimum baseline":float(_a["baseline"][7:].min()),"Minimum observed sales":float(_a["observed"][7:].min())})
    tv_advi_summary=[]
    for _r in tv_advi_reports:
        _g=next(g for g in _r["gates"] if g["relative"]==.1 and g["absolute"]==.01)
        tv_advi_summary.append({"World":_r["world_id"]+1,"Method":_r["method"],"Media point pass":_g["media_point_pass"],"Total media point pass":_g["total_media_point_pass"],
            "Interval width / NUTS":_r["median_interval_width_ratio_to_nuts"],"Mean gap / NUTS width":_r["median_mean_gap_in_nuts_widths"],
            "Last loss block":_r["last_1000_mean_loss"],"Previous loss block":_r["previous_1000_mean_loss"]})
    write_study_json(StudyPath("notebooks/time_varying_baseline/comparison.json"),{"paired":tv_paired_summary,"overall":tv_overall_summary,"patterns":tv_pattern_summary,"realism_flags":tv_realism_flags,"advi":tv_advi_summary})
    mo.vstack([mo.md("## Time-varying baseline extension: matched 50-world study\nAll 50 Nutpie draw-diag fits pass the prespecified computational screen. Priors match the generator: baseline level Uniform(3,8), relative walk amplitude Uniform(0,.093) × media-coefficient norm, smoothing Beta(2,2) mapped to widths 1–26. The centered/smoothed walk spans **112 weeks**, with the same burn-in and fitted-week handling as generation. The path and width are marginalized exactly, then conditional baseline paths are sampled. Neither true baseline hyperparameters nor the true path are supplied to fitting.\nThe original inputs, graph, continuous effects, constant baseline level and observation-noise realization are preserved. Original results remain untouched. Recovery below uses the **10% / 1%** gate."),mo.ui.table(tv_paired_summary,selection=None),mo.ui.table(tv_overall_summary,selection=None),
        mo.md("### Pattern associations\nFlags come from saved generation labels and are **not exclusive**. Counts concern nonzero-effect media. They are descriptive, not causal: input count, signal-to-noise, saturation and multiple simultaneous patterns can explain apparent associations. Ramping-up worlds were already difficult with constant baselines. No all-off contributing channel occurred in this sample, so that case remains untested."),mo.ui.table(tv_pattern_summary,selection=None),
        mo.md("### ADVI pilot\nFive worlds, two seeds, full-rank ADVI, 20,000 steps, 4,000 draws. Losses continue improving in several fits; interval and mean discrepancies remain. Agreement on binary recovery does not validate ADVI as a screening method—none of these five worlds passes the MCMC individual-media gate. No MCMC diagnostics are applied to variational draws."),mo.ui.table(tv_advi_summary,selection=None),
        mo.md("### Realism flags\nThe configured generator has no baseline floor. A negative intrinsic baseline can matter if it is interpreted as nonnegative organic sales; it is not automatically invalid for a general additive model. This is an empirical realism check, separate from sampling diagnostics."),mo.ui.table(tv_realism_flags,selection=None)])

    return


@app.cell(hide_code=True)
def tv_baseline_scorecard_controls(mo):
    tv_view_world=mo.ui.dropdown(options={f"Matched world {w+1}":w for w in range(50)},value="Matched world 1",label="Baseline comparison world")
    tv_view_regime=mo.ui.dropdown(options={"Time-varying baseline":"varying","Constant baseline reference":"constant"},value="Time-varying baseline",label="Baseline regime")
    tv_view_advi=mo.ui.dropdown(options={"Off":None,"ADVI seed 1":"fullrank_advi_1","ADVI seed 2":"fullrank_advi_2"},value="Off",label="ADVI overlay (time-varying worlds 1–5)")
    mo.hstack([tv_view_world,tv_view_regime,tv_view_advi])

    return tv_view_advi, tv_view_regime, tv_view_world


@app.cell(hide_code=True)
def tv_baseline_scorecard(
    StudyPath,
    bayes_xr,
    bayesian_reports,
    go,
    make_subplots,
    mo,
    np,
    pg,
    saved_study_records,
    scorecard_geometry_by_world,
    tv_full_reports,
    tv_patterns,
    tv_view_advi,
    tv_view_regime,
    tv_view_world,
):
    _w=tv_view_world.value;_varying=tv_view_regime.value=="varying"
    _path=StudyPath("notebooks/time_varying_baseline/nuts")/f"world_{_w:02d}_nuts_recovery.npz" if _varying else StudyPath("notebooks/bayesian_mmm_draw_diag")/f"world_{_w:02d}_recovery.npz"
    with np.load(_path) as _f:_a={k:_f[k] for k in _f.files}
    if not _varying:
        with np.load(StudyPath("notebooks/total_media_recovery")/f"world_{_w:02d}_total_media.npz") as _f:
            _a.update(total_media_truth=_f["truth"],total_media_mean=_f["nuts_mean"],total_media_lower95=_f["nuts_lower95"],total_media_upper95=_f["nuts_upper95"])
        with bayes_xr.open_datatree(StudyPath("notebooks/bayesian_mmm_draw_diag")/f"world_{_w:02d}_posterior.nc",engine="h5netcdf") as _f:
            _a["expected_lower95"],_a["expected_upper95"]=np.quantile(_f["posterior"]["mu"].values.reshape(-1,97),[.025,.975],axis=0)
    _advi=None
    if _varying and _w<5 and tv_view_advi.value is not None:
        with np.load(StudyPath("notebooks/time_varying_baseline/advi")/f"world_{_w:02d}_{tv_view_advi.value}_recovery.npz") as _f:_advi={k:_f[k] for k in _f.files}
    _record=saved_study_records[_w];_names=_record["terms"]
    _indices=[0]+scorecard_geometry_by_world[_w]["indices"]
    _titles=["Observed sales and expected outcome","Total media contribution"]+["Baseline path" if i==0 else _names[i] for i in _indices]
    _corpus=pg.load_corpus(StudyPath("notebooks/ols_study_50_seed10029")/f"batch_{_record['batch']:02d}.npz")
    _nrows=len(_titles)
    _fig=make_subplots(rows=_nrows,cols=1,shared_xaxes=True,vertical_spacing=.24/_nrows,subplot_titles=_titles,
        specs=[[{}],[{}]]+[[{"secondary_y":_names[i].startswith("C")}] for i in _indices])
    _weeks=np.arange(7,104)
    def _band(row,lower,upper,label,fill,legend=False):
        _fig.add_trace(go.Scatter(x=_weeks,y=lower,mode="lines",line={"width":0},showlegend=False,hoverinfo="skip",legendgroup=label),row=row,col=1)
        _fig.add_trace(go.Scatter(x=_weeks,y=upper,mode="lines",line={"width":0},fill="tonexty",fillcolor=fill,name=label,legendgroup=label,showlegend=legend,hoverinfo="skip"),row=row,col=1)
    def _line(row,y,label,color,dash="solid",legend=False,secondary=False):
        _fig.add_trace(go.Scatter(x=_weeks,y=y,mode="lines",line={"color":color,"dash":dash},name=label,legendgroup=label,showlegend=legend),row=row,col=1,secondary_y=secondary)
    _band(1,_a["predictive_lower95"],_a["predictive_upper95"],"NUTS predictive 95%","rgba(255,140,0,.1)",True)
    _band(1,_a["expected_lower95"],_a["expected_upper95"],"NUTS expected sales 95%","rgba(255,140,0,.3)",True)
    _line(1,_a["noiseless"],"Truth","black",legend=True)
    _line(1,_a["mean"].sum(axis=1),"NUTS mean","darkorange",legend=True)
    _fig.add_trace(go.Scatter(x=_weeks,y=_a["observed"],mode="markers",marker={"color":"gray","size":4},name="Observed sales"),row=1,col=1)
    _band(2,_a["total_media_lower95"],_a["total_media_upper95"],"NUTS contribution 95%","rgba(255,140,0,.2)",True)
    _line(2,_a["total_media_truth"],"Truth","black")
    _line(2,_a["total_media_mean"],"NUTS mean","darkorange")
    if _advi is not None:
        _band(1,_advi["expected_lower95"],_advi["expected_upper95"],"ADVI expected sales 95%","rgba(148,103,189,.18)",True)
        _line(1,_advi["mean"].sum(axis=1),"ADVI mean","#9467bd","dot",True)
        _band(2,_advi["total_media_lower95"],_advi["total_media_upper95"],"ADVI contribution 95%","rgba(148,103,189,.18)",True)
        _line(2,_advi["total_media_mean"],"ADVI mean","#9467bd","dot")
    _spend_legend=True
    for _row,_i in enumerate(_indices,start=3):
        _band(_row,_a["lower95"][:,_i],_a["upper95"][:,_i],"NUTS contribution 95%","rgba(255,140,0,.2)")
        _line(_row,_a["truth"][:,_i],"Truth","black")
        _line(_row,_a["mean"][:,_i],"NUTS mean","darkorange")
        if _advi is not None:
            _band(_row,_advi["lower95"][:,_i],_advi["upper95"][:,_i],"ADVI contribution 95%","rgba(148,103,189,.18)")
            _line(_row,_advi["mean"][:,_i],"ADVI mean","#9467bd","dot")
        if _names[_i].startswith("C"):
            _line(_row,_corpus["treatment_raw"][_record["local_world"],7:,int(_names[_i][1:])-1],"Raw spend (right)","#888888","dot",_spend_legend,True)
            _spend_legend=False
            _fig.update_yaxes(title_text="Raw spend",showgrid=False,rangemode="tozero",row=_row,col=1,secondary_y=True)
    _fig.update_xaxes(showticklabels=True,title_text="Week")
    _fig.update_layout(height=320*_nrows+160,template="plotly_white",title=f"World {_w+1} · {'time-varying' if _varying else 'constant'} baseline",margin={"t":100,"b":140},legend={"orientation":"h","y":-.012,"yanchor":"top"})
    _report=tv_full_reports[_w] if _varying else bayesian_reports[_w]
    _patterns=[p for p in tv_patterns if p["world_id"]==_w and p["nonzero_effect"]]
    mo.vstack([mo.md(f"**Diagnostics pass: {_report['diagnostics_pass']}** · divergences {_report['divergences']} · R-hat {_report['max_rhat']:.4f}. All bands are 95% pointwise equal-tail intervals. Baseline and total-media uncertainty preserve joint posterior dependence. This is in-sample reconstruction, not a forecast. ADVI is unavailable outside the first five time-varying worlds."),mo.ui.plotly(_fig),mo.ui.table(_patterns,selection=None)])

    return


@app.cell(hide_code=True)
def tv_baseline_final_validation(
    StudyPath,
    bayes_xr,
    mo,
    np,
    study_hashlib,
    study_json,
    tv_bases,
    tv_records,
    tv_world_data,
    write_study_json,
):
    tv_final_validation={"matched_worlds_checked":0,"posterior_outputs_checked":0,"source_batches_unchanged":0,"original_posterior_hashes_unchanged":0}
    _manifest=study_json.loads(StudyPath("notebooks/time_varying_baseline/data_manifest.json").read_text())
    for _name,_sha in _manifest["source_sha256"].items():
        assert study_hashlib.sha256((StudyPath("notebooks/ols_study_50_seed10029")/_name).read_bytes()).hexdigest()==_sha
        tv_final_validation["source_batches_unchanged"]+=1
    _original_hashes=study_json.loads(StudyPath("notebooks/bayesian_mmm_draw_diag/extension_provenance.json").read_text())["reused_posterior_sha256"]
    for _name,_sha in _original_hashes.items():
        assert study_hashlib.sha256((StudyPath("notebooks/bayesian_mmm_draw_diag")/_name).read_bytes()).hexdigest()==_sha
        tv_final_validation["original_posterior_hashes_unchanged"]+=1
    for _record in tv_records:
        _w=_record["world_id"];_local=_record["local_world"]
        _data,_corpus=tv_world_data(_record)
        with np.load(StudyPath("notebooks/time_varying_baseline")/f"world_{_w:02d}_data.npz") as _generated:
            _old_baseline=_corpus["baseline_intrinsic"][_local].astype(float)
            np.testing.assert_allclose(_generated["observed"]-_generated["baseline"],_corpus["outcome_raw"][_local].astype(float)-_old_baseline,atol=1e-12)
            np.testing.assert_allclose(_generated["baseline_full"],_old_baseline[0]+_record["true_baseline_sd"]*tv_bases[_record["true_width"]-1]@_generated["innovations"],rtol=1e-12,atol=1e-12)
            tv_final_validation["matched_worlds_checked"]+=1
        _methods=[("nuts","nuts")]+([("advi","fullrank_advi_1"),("advi","fullrank_advi_2")] if _w<5 else [])
        for _folder,_method in _methods:
            _directory=StudyPath("notebooks/time_varying_baseline")/_folder
            with bayes_xr.open_datatree(_directory/f"world_{_w:02d}_{_method}_posterior.nc",engine="h5netcdf") as _tree:
                _p=_tree["posterior"]
                for _name,_lo,_hi in (("intercept",3,8),("amplitude",.5,2),("alpha",.2,.8),("mm_kappa_mult",.05,20),("control_reference_contribution",-.3,.3),("sigma_relative",.01,.028),("baseline_sd_relative",0,.093)):
                    _values=_p[_name].values
                    assert np.isfinite(_values).all() and np.all((_values>=_lo-1e-10)&(_values<=_hi+1e-10))
                _norm=np.linalg.norm(_p["beta"].values,axis=-1)
                np.testing.assert_allclose(_p["sigma"].values,_p["sigma_relative"].values*_norm,rtol=1e-10,atol=1e-10)
                np.testing.assert_allclose(_p["baseline_sd"].values,_p["baseline_sd_relative"].values*_norm,rtol=1e-10,atol=1e-10)
                if _folder=="nuts":
                    np.testing.assert_allclose(_tree["observed_data"]["outcome_raw"].values,_data["outcome_data"]*_data["outcome_scale"],rtol=1e-12,atol=1e-12)
                with bayes_xr.open_dataset(_directory/f"world_{_w:02d}_{_method}_baseline_draws.nc",engine="h5netcdf") as _derived:
                    assert np.all((_derived["width"].values>=1)&(_derived["width"].values<=26))
                    with np.load(_directory/f"world_{_w:02d}_{_method}_recovery.npz") as _a,np.load(StudyPath("notebooks/ols_study_50_seed10029")/f"world_{_w:02d}_fits.npz") as _old:
                        np.testing.assert_array_equal(_a["truth"][:,1:],_old["truth"][:,1:])
                        np.testing.assert_allclose(_a["mean"][:,0],_derived["baseline_path"].values.reshape(-1,97).mean(axis=0),atol=1e-12)
                        _full_mu=_p["mu"].values+_derived["baseline_path"].values-_p["intercept"].values[:,:,None]
                        np.testing.assert_allclose(_a["mean"].sum(axis=1),_full_mu.reshape(-1,97).mean(axis=0),rtol=1e-10,atol=1e-10)
                        assert all(np.isfinite(_a[k]).all() for k in _a.files)
                        assert np.all(_a["lower95"]<=_a["upper95"])
                        _null=np.flatnonzero(~np.any(_old["truth"]!=0,axis=0))
                        assert np.all(_a["mean"][:,_null]==0) and np.all(_a["upper95"][:,_null]==0)
            tv_final_validation["posterior_outputs_checked"]+=1
    write_study_json(StudyPath("notebooks/time_varying_baseline/final_validation.json"),tv_final_validation)
    mo.md(f"Time-varying-baseline validation passed: {tv_final_validation['matched_worlds_checked']} matched worlds, {tv_final_validation['posterior_outputs_checked']} inference outputs, original source batches and original five posterior hashes unchanged.")

    return


@app.cell(hide_code=True)
def tv_media_pass_fail_profile(
    SlotLayout,
    StudyPath,
    np,
    pg,
    saved_study_records,
    transform_media_columns,
    tv_full_reports,
    write_study_json,
):
    def profile_media_pass_fail():
        rows=[]
        for record in saved_study_records:
            w=record["world_id"];local=record["local_world"]
            report=tv_full_reports[w]
            gate=next(g for g in report["gates"] if g["relative"]==.1 and g["absolute"]==.01)
            corpus=pg.load_corpus(StudyPath("notebooks/ols_study_50_seed10029")/f"batch_{record['batch']:02d}.npz")
            graph=SlotLayout(15,10,1).unpack(corpus["g"][local])
            slots=np.flatnonzero(graph["cy"])
            x=corpus["treatment_raw"][local][:,slots].astype(float)
            alpha=corpus["carryover_alpha"][local,slots].astype(float)
            kappa=corpus["mm_kappa_mult"][local,slots].astype(float)
            anchor=corpus["mechanism_saturation_scale"][local,slots].astype(float)
            response=transform_media_columns(x,alpha,kappa*anchor)[7:]
            carried=np.column_stack([np.convolve(x[:,j],alpha[j]**np.arange(8)/np.sum(alpha[j]**np.arange(8)))[:104] for j in range(len(slots))])[7:]
            np.testing.assert_allclose(response,carried/(carried+kappa*anchor),rtol=1e-10,atol=1e-10)
            noise_sd=float(np.std(corpus["outcome_noise"][local,7:].astype(float)))
            with np.load(StudyPath("notebooks/time_varying_baseline/nuts")/f"world_{w:02d}_nuts_recovery.npz") as fitted:
                truth=fitted["truth"].copy();estimate=fitted["mean"].copy()
            for j,slot in enumerate(slots):
                channel=f"C{slot+1}";position=record["terms"].index(channel)
                contribution=truth[:,position];raw=x[7:,j]
                raw_cv=float(raw.std()/raw.mean()) if raw.mean()!=0 else None
                carried_cv=float(carried[:,j].std()/carried[:,j].mean()) if carried[:,j].mean()!=0 else None
                contribution_cv=float(contribution.std()/abs(contribution.mean())) if contribution.mean()!=0 else None
                allowance=max(.1*np.linalg.norm(contribution),.01*np.linalg.norm(truth.sum(axis=1)))
                ratio=float(np.linalg.norm(estimate[:,position]-contribution)/allowance)
                failed=channel in gate["failing_channels"]
                assert failed==(ratio>1)
                rows.append({"World":w+1,"Channel":channel,"MCMC valid":report["diagnostics_pass"],"Failed":failed,"Error / tolerance":ratio,
                    "Mean saturation":float(response[:,j].mean()),"Median saturation":float(np.median(response[:,j])),"Maximum saturation":float(response[:,j].max()),
                    "Fraction weeks ≥95% saturation":float(np.mean(response[:,j]>=.95)),"Fraction weeks ≤50% saturation":float(np.mean(response[:,j]<=.5)),
                    "Always ≤50% saturation":bool(response[:,j].max()<=.5),"Adstock alpha":float(alpha[j]),"Kappa multiplier":float(kappa[j]),
                    "Beta (asymptotic contribution)":float(corpus["beta"][local,slot]),"Reference contribution amplitude":float(corpus["treatment_reference_contribution"][local,slot]),
                    "Raw spend CV":raw_cv,"Adstocked spend CV":carried_cv,"Contribution CV":contribution_cv,
                    "CV retained by adstock":carried_cv/raw_cv if raw_cv else None,"CV retained by saturation":contribution_cv/carried_cv if carried_cv else None,
                    "Mean contribution":float(contribution.mean()),"Contribution temporal SD":float(contribution.std()),"Realized noise SD":noise_sd,
                    "Contribution SD / noise SD":float(contribution.std()/noise_sd),"Media effects in world":len(slots)})
        write_study_json(StudyPath("notebooks/time_varying_baseline/media_pass_fail_profiles.json"),rows)
        return rows

    tv_media_pass_fail_profiles=profile_media_pass_fail()

    return (tv_media_pass_fail_profiles,)


@app.cell(hide_code=True)
def tv_media_pass_fail_comparison(
    StudyPath,
    go,
    make_subplots,
    mo,
    np,
    tv_media_pass_fail_profiles,
    write_study_json,
):
    _rows=[r for r in tv_media_pass_fail_profiles if r["MCMC valid"]]
    _features=["Mean saturation","Maximum saturation","Adstock alpha","Kappa multiplier","Beta (asymptotic contribution)","Reference contribution amplitude","Raw spend CV","Adstocked spend CV","Contribution CV","CV retained by adstock","CV retained by saturation","Contribution temporal SD","Realized noise SD","Contribution SD / noise SD"]
    tv_media_parameter_comparison=[]
    for _feature in _features:
        _item={"True quantity":_feature}
        for _label,_status in (("Pass",False),("Fail",True)):
            _values=[r[_feature] for r in _rows if r["Failed"]==_status and r[_feature] is not None]
            _q=np.quantile(_values,[.25,.5,.75])
            _item[_label+" median"]=float(_q[1]);_item[_label+" Q1"]=float(_q[0]);_item[_label+" Q3"]=float(_q[2])
        tv_media_parameter_comparison.append(_item)
    tv_media_failure_bins=[]
    for _feature,_edges in (("Mean saturation",[0,.25,.5,.75,1.00001]),("Contribution CV",[0,.1,.25,.5,float("inf")]),("Contribution SD / noise SD",[0,.5,1,2,float("inf")]),("Adstock alpha",[.2,.4,.6,.8])):
        for _i,(_lo,_hi) in enumerate(zip(_edges[:-1],_edges[1:])):
            _subset=[r for r in _rows if _lo<=r[_feature] and (r[_feature]<_hi or (_i==len(_edges)-2 and r[_feature]==_hi))]
            _failed=sum(r["Failed"] for r in _subset)
            tv_media_failure_bins.append({"Quantity":_feature,"Bin":f"[{_lo:g}, {_hi:g}{']' if _i==len(_edges)-2 else ')'}","Channels":len(_subset),"Failures":_failed,"Failure rate":_failed/len(_subset) if _subset else None})
    _mixed=[w for w in range(1,51) if any(r["World"]==w and r["Failed"] for r in _rows) and any(r["World"]==w and not r["Failed"] for r in _rows)]
    tv_media_within_world=[]
    for _feature in ("Mean saturation","Adstock alpha","Beta (asymptotic contribution)","Reference contribution amplitude","Raw spend CV","Contribution CV","Contribution SD / noise SD"):
        _delta=[float(np.median([r[_feature] for r in _rows if r["World"]==w and r["Failed"]])-np.median([r[_feature] for r in _rows if r["World"]==w and not r["Failed"]])) for w in _mixed]
        tv_media_within_world.append({"Quantity":_feature,"Mixed worlds":len(_mixed),"Median within-world fail minus pass":float(np.median(_delta)),"Worlds with lower median among failures":sum(d<0 for d in _delta)})
    _fig=make_subplots(rows=2,cols=3,subplot_titles=("Mean saturation","Adstock alpha","Asymptotic beta","Raw spend CV","Contribution CV","Contribution SD / noise SD"),vertical_spacing=.22,horizontal_spacing=.12)
    for _i,_feature in enumerate(("Mean saturation","Adstock alpha","Beta (asymptotic contribution)","Raw spend CV","Contribution CV","Contribution SD / noise SD")):
        for _label,_status,_color in (("Pass",False,"#2ca02c"),("Fail",True,"#d62728")):
            _subset=[r for r in _rows if r["Failed"]==_status]
            _fig.add_trace(go.Box(y=[r[_feature] for r in _subset],name=_label,legendgroup=_label,showlegend=_i==0,marker_color=_color,boxpoints="all",jitter=.35,pointpos=0,
                text=[f"World {r['World']} · {r['Channel']} · error/tolerance {r['Error / tolerance']:.2f}" for r in _subset]),row=_i//3+1,col=_i%3+1)
        if _feature in ("Beta (asymptotic contribution)","Contribution SD / noise SD"):_fig.update_yaxes(type="log",row=_i//3+1,col=_i%3+1)
    _fig.update_layout(height=780,template="plotly_white",title="True properties of passing versus failing media channels",legend={"orientation":"h","y":-.12,"yanchor":"top"},margin={"b":110})
    write_study_json(StudyPath("notebooks/time_varying_baseline/media_pass_fail_summary.json"),{"medians_and_iqr":tv_media_parameter_comparison,"failure_bins":tv_media_failure_bins,"within_world":tv_media_within_world})
    mo.vstack([mo.md(f"### Which media channels fail?\n**{sum(r['Failed'] for r in _rows)} failures / {len(_rows)} nonzero media channels**, time-varying-baseline MCMC, 10% / 1% gate. All properties below are **true generator quantities**, not failed-fit parameter estimates. Realized saturation is the unit response after adstock, averaged over weeks 7–103. Beta is the asymptotic contribution, not the reference-input amplitude. CV is temporal SD / mean; the noise denominator is realized observation-noise SD, not baseline variation.\nRows are channels, not independent datasets. Pooled medians weight worlds with more channels more heavily. Within-world comparisons below reduce between-world confounding but remain descriptive, not causal."),mo.ui.plotly(_fig),mo.ui.table(tv_media_parameter_comparison,selection=None),mo.ui.table(tv_media_failure_bins,selection=None),mo.ui.table(tv_media_within_world,selection=None),mo.ui.table(_rows,selection=None)])

    return


@app.cell(hide_code=True)
def media_parameter_tsne(
    StudyPath,
    np,
    tv_media_pass_fail_profiles,
    tv_thread_limits,
    write_study_json,
):
    from sklearn.manifold import TSNE as MediaTSNE, trustworthiness as media_tsne_trustworthiness
    from sklearn.preprocessing import StandardScaler as MediaStandardScaler
    from sklearn.neighbors import NearestNeighbors as MediaNearestNeighbors
    from sklearn.decomposition import PCA as MediaPCA

    media_tsne_rows=[r for r in tv_media_pass_fail_profiles if r["MCMC valid"]]
    # No recovery errors, fitted values, or pass/fail labels enter the embedding.
    media_tsne_feature_names=["Adstock alpha","log kappa","log beta","Reference amplitude","Mean saturation","log1p raw-spend CV","log1p adstocked-spend CV","log1p contribution CV","log contribution variance","log observation-noise variance"]
    media_tsne_raw_features=np.array([[r["Adstock alpha"],np.log(r["Kappa multiplier"]),np.log(r["Beta (asymptotic contribution)"]),r["Reference contribution amplitude"],r["Mean saturation"],
        np.log1p(r["Raw spend CV"]),np.log1p(r["Adstocked spend CV"]),np.log1p(r["Contribution CV"]),np.log(max(r["Contribution temporal SD"]**2,1e-16)),np.log(max(r["Realized noise SD"]**2,1e-16))] for r in media_tsne_rows])
    media_tsne_features=MediaStandardScaler().fit_transform(media_tsne_raw_features)
    media_tsne_labels=np.array([r["Failed"] for r in media_tsne_rows],dtype=bool)
    media_tsne_worlds=np.array([r["World"] for r in media_tsne_rows])
    media_tsne_embeddings={}
    with tv_thread_limits(limits=1):
        for _perplexity in (15,30,50):
            for _seed in (2026,2027,2028):
                media_tsne_embeddings[f"p{_perplexity}_seed{_seed}"]=MediaTSNE(n_components=2,perplexity=_perplexity,init="random",learning_rate="auto",max_iter=1500,random_state=_seed).fit_transform(media_tsne_features)
        media_tsne_pca=MediaPCA(n_components=2).fit_transform(media_tsne_features)
    # A second feature set removes deterministic beta/reference-amplitude redundancy
    # and intermediate adstock CV; use this only as an exploratory robustness view.
    media_tsne_reduced_features=MediaStandardScaler().fit_transform(media_tsne_raw_features[:,[0,1,3,4,5,7,8,9]])
    with tv_thread_limits(limits=1):
        media_tsne_reduced=MediaTSNE(n_components=2,perplexity=30,init="random",learning_rate="auto",max_iter=1500,random_state=2026).fit_transform(media_tsne_reduced_features)


    def media_neighbor_mixing(features,labels,worlds,k=10):
        # Exclude the same world: otherwise shared noise and context inflate similarity.
        order=MediaNearestNeighbors(n_neighbors=len(labels)).fit(features).kneighbors(features,return_distance=False)
        neighbors=np.array([[j for j in order[i] if worlds[j]!=worlds[i]][:k] for i in range(len(labels))])
        local_fail=labels[neighbors].mean(axis=1)
        return {"Mean neighboring failure fraction around passed channels":float(local_fail[~labels].mean()),
            "Mean neighboring failure fraction around failed channels":float(local_fail[labels].mean()),
            "Same-label neighbor fraction":float((labels[neighbors]==labels[:,None]).mean())}

    media_tsne_neighborhood_summary=[]
    for _name,_embedding in [("Original standardized features",media_tsne_features),("PCA",media_tsne_pca),("Reduced-feature t-SNE",media_tsne_reduced)]+list(media_tsne_embeddings.items()):
        media_tsne_neighborhood_summary.append({"Representation":_name,**media_neighbor_mixing(_embedding,media_tsne_labels,media_tsne_worlds),
            "Trustworthiness (10 neighbors)":float(media_tsne_trustworthiness(media_tsne_reduced_features if _name=="Reduced-feature t-SNE" else media_tsne_features,_embedding,n_neighbors=10))})
    _tsne_dir=StudyPath("notebooks/time_varying_baseline/channel_tsne");_tsne_dir.mkdir(exist_ok=True)
    np.savez_compressed(_tsne_dir/"embeddings.npz",features=media_tsne_features,failed=media_tsne_labels,world=media_tsne_worlds,pca=media_tsne_pca,reduced_tsne=media_tsne_reduced,**media_tsne_embeddings)
    write_study_json(_tsne_dir/"metadata.json",{"initialization":"random (independent seeded starts)","feature_names":media_tsne_feature_names,"neighborhood_summary":media_tsne_neighborhood_summary,
        "rows":[{"World":r["World"],"Channel":r["Channel"],"Failed":r["Failed"]} for r in media_tsne_rows],"note":"Unsupervised exploratory embeddings. No errors or labels used for fitting. Neighbor summaries exclude same-world channels; they are not held-out predictive validation. Correlated features can overweight common information."})

    return (
        MediaStandardScaler,
        media_tsne_embeddings,
        media_tsne_feature_names,
        media_tsne_labels,
        media_tsne_neighborhood_summary,
        media_tsne_raw_features,
        media_tsne_rows,
        media_tsne_worlds,
    )


@app.cell(hide_code=True)
def media_parameter_tsne_controls(mo):
    media_tsne_perplexity=mo.ui.dropdown(options={"15":15,"30":30,"50":50},value="30",label="t-SNE perplexity")
    media_tsne_seed=mo.ui.dropdown(options={"2026":2026,"2027":2027,"2028":2028},value="2026",label="t-SNE seed")
    mo.hstack([media_tsne_perplexity,media_tsne_seed])

    return media_tsne_perplexity, media_tsne_seed


@app.cell(hide_code=True)
def media_parameter_tsne_view(
    go,
    media_tsne_embeddings,
    media_tsne_labels,
    media_tsne_neighborhood_summary,
    media_tsne_perplexity,
    media_tsne_rows,
    media_tsne_seed,
    mo,
    np,
):
    _embedding=media_tsne_embeddings[f"p{media_tsne_perplexity.value}_seed{media_tsne_seed.value}"]
    _fig=go.Figure()
    for _failed,_label,_color in ((False,"Pass (92)","#0072B2"),(True,"Fail (144)","#D55E00")):
        _ix=np.flatnonzero(media_tsne_labels==_failed)
        _fig.add_trace(go.Scatter(x=_embedding[_ix,0],y=_embedding[_ix,1],mode="markers",name=_label,
            marker={"color":_color,"size":9,"opacity":.8,"symbol":"circle" if _failed else "diamond"},
            text=[f"World {media_tsne_rows[i]['World']} · {media_tsne_rows[i]['Channel']}<br>Mean saturation {media_tsne_rows[i]['Mean saturation']:.1%}<br>Alpha {media_tsne_rows[i]['Adstock alpha']:.2f}<br>Beta {media_tsne_rows[i]['Beta (asymptotic contribution)']:.2f}<br>Contribution CV {media_tsne_rows[i]['Contribution CV']:.1%}<br>Contribution SD/noise SD {media_tsne_rows[i]['Contribution SD / noise SD']:.2f}" for i in _ix],hovertemplate="%{text}<extra>%{fullData.name}</extra>"))
    _fig.update_layout(height=650,template="plotly_white",title="Media-channel t-SNE: true parameters, variation and noise",xaxis_title="t-SNE 1 (arbitrary)",yaxis_title="t-SNE 2 (arbitrary)",legend={"orientation":"h","y":-.15,"yanchor":"top"})
    mo.vstack([mo.md("### Exploratory channel embedding\nOne point per nonzero media channel; time-varying-baseline MCMC, 10% / 1% gate. Parameters and variation/noise features are log-transformed where appropriate, then standardized. **Labels and recovery errors are not embedding inputs.** Features include alpha, kappa, beta, reference amplitude, mean saturation, raw/adstocked/contribution CV, contribution variance and observation-noise variance. Beta is algebraically related to reference amplitude and kappa; a reduced-feature sensitivity embedding is saved too.\nUse the seed/perplexity controls to check stability. t-SNE emphasizes local neighborhoods: island appearance, axis direction and distances between islands are not evidence of distinct populations. Channels share world-level context; the neighbor table excludes neighbors from the same world."),mo.ui.plotly(_fig),mo.ui.table(media_tsne_neighborhood_summary,selection=None)])

    return


@app.cell(hide_code=True)
def media_failure_logistic_regression(
    MediaStandardScaler,
    StudyPath,
    media_tsne_feature_names,
    media_tsne_labels,
    media_tsne_raw_features,
    media_tsne_worlds,
    np,
    tv_thread_limits,
    write_study_json,
):
    from sklearn.linear_model import LogisticRegression as MediaLogisticRegression
    from sklearn.pipeline import make_pipeline as media_make_pipeline
    from sklearn.model_selection import GroupKFold as MediaGroupKFold
    from sklearn.metrics import roc_auc_score as media_roc_auc, average_precision_score as media_average_precision, log_loss as media_log_loss, brier_score_loss as media_brier_score, balanced_accuracy_score as media_balanced_accuracy, confusion_matrix as media_confusion_matrix
    from sklearn.calibration import calibration_curve as media_calibration_curve

    # All transformations before this point are fixed, row-wise log transforms.
    # Standardization is learned anew INSIDE each training fold.
    _media_X=media_tsne_raw_features
    _media_y=media_tsne_labels.astype(int)
    _media_groups=media_tsne_worlds
    media_logistic_oof=np.full(len(_media_y),np.nan)
    media_logistic_baseline_oof=np.full(len(_media_y),np.nan)
    media_logistic_fold_ids=np.full(len(_media_y),-1)
    media_logistic_folds=[]
    media_logistic_fold_coefficients=[]
    with tv_thread_limits(limits=1):
        for _fold,(_train,_test) in enumerate(MediaGroupKFold(n_splits=5,shuffle=True,random_state=2026).split(_media_X,_media_y,_media_groups)):
            assert not set(_media_groups[_train])&set(_media_groups[_test])
            _model=media_make_pipeline(MediaStandardScaler(),MediaLogisticRegression(C=1.0,solver="lbfgs",max_iter=5000))
            _model.fit(_media_X[_train],_media_y[_train])
            assert _model[-1].n_iter_.max()<5000
            media_logistic_oof[_test]=_model.predict_proba(_media_X[_test])[:,1]
            media_logistic_baseline_oof[_test]=_media_y[_train].mean()
            media_logistic_fold_ids[_test]=_fold
            media_logistic_fold_coefficients.append(_model[-1].coef_[0].copy())
            media_logistic_folds.append({"Fold":_fold+1,"Test worlds":sorted(set(int(w) for w in _media_groups[_test])),"Channels":len(_test),"Failure fraction":float(_media_y[_test].mean()),
                "ROC AUC":float(media_roc_auc(_media_y[_test],media_logistic_oof[_test])),"Brier score":float(media_brier_score(_media_y[_test],media_logistic_oof[_test]))})
    assert np.isfinite(media_logistic_oof).all() and np.all(media_logistic_fold_ids>=0)
    media_logistic_scores=[]
    for _label,_prob in (("Logistic: held-out-world predictions",media_logistic_oof),("Training-fold prevalence baseline",media_logistic_baseline_oof)):
        _prediction=_prob>=.5
        _tn,_fp,_fn,_tp=media_confusion_matrix(_media_y,_prediction).ravel()
        media_logistic_scores.append({"Model":_label,"ROC AUC":float(media_roc_auc(_media_y,_prob)),"Average precision":float(media_average_precision(_media_y,_prob)),
            "Log loss":float(media_log_loss(_media_y,_prob)),"Brier score":float(media_brier_score(_media_y,_prob)),"Balanced accuracy at 0.5":float(media_balanced_accuracy(_media_y,_prediction)),
            "Failure sensitivity":float(_tp/(_tp+_fn)),"Pass specificity":float(_tn/(_tn+_fp)),"TN":int(_tn),"FP":int(_fp),"FN":int(_fn),"TP":int(_tp)})
    # Fit all channels only for a descriptive coefficient table and future use.
    # Its in-sample score is deliberately not reported as predictive performance.
    media_logistic_model=media_make_pipeline(MediaStandardScaler(),MediaLogisticRegression(C=1.0,solver="lbfgs",max_iter=5000))
    with tv_thread_limits(limits=1):media_logistic_model.fit(_media_X,_media_y)
    media_logistic_coefficients=[]
    for _i,_name in enumerate(media_tsne_feature_names):
        _b=float(media_logistic_model[-1].coef_[0,_i]);_fold_b=np.array(media_logistic_fold_coefficients)[:,_i]
        media_logistic_coefficients.append({"Feature":_name,"Standardized coefficient":_b,"Failure odds ratio per feature SD":float(np.exp(_b)),
            "Fold minimum coefficient":float(_fold_b.min()),"Fold maximum coefficient":float(_fold_b.max())})
    # Cluster-bootstrap the already out-of-fold predictions by world, not channel.
    # These intervals describe uncertainty in this OOF evaluation, not a fresh external test.
    _rng=np.random.default_rng(2027);_world_ids=np.unique(_media_groups);_bootstrap=[]
    for _rep in range(1000):
        _picked=_rng.choice(_world_ids,size=len(_world_ids),replace=True)
        _ix=np.concatenate([np.flatnonzero(_media_groups==w) for w in _picked])
        if len(np.unique(_media_y[_ix]))==2:
            _bootstrap.append([media_roc_auc(_media_y[_ix],media_logistic_oof[_ix]),media_brier_score(_media_y[_ix],media_logistic_oof[_ix])-media_brier_score(_media_y[_ix],media_logistic_baseline_oof[_ix])])
    media_logistic_uncertainty={"OOF AUC world-bootstrap 95% interval":np.quantile(np.array(_bootstrap)[:,0],[.025,.975]).tolist(),"OOF Brier difference vs baseline 95% interval":np.quantile(np.array(_bootstrap)[:,1],[.025,.975]).tolist(),
        "note":"World-cluster bootstrap of fixed cross-validated predictions; does not refit models or include all training/split uncertainty."}
    _logistic_dir=StudyPath("notebooks/time_varying_baseline/channel_logistic");_logistic_dir.mkdir(exist_ok=True)
    np.savez_compressed(_logistic_dir/"model_and_predictions.npz",oof_probability=media_logistic_oof,baseline_probability=media_logistic_baseline_oof,failed=_media_y,world=_media_groups,fold=media_logistic_fold_ids,
        coefficient=media_logistic_model[-1].coef_,intercept=media_logistic_model[-1].intercept_,scaler_mean=media_logistic_model[0].mean_,scaler_scale=media_logistic_model[0].scale_)
    write_study_json(_logistic_dir/"summary.json",{"features":media_tsne_feature_names,"penalty":"L2, C=1, no tuning or class weighting","folds":media_logistic_folds,"scores":media_logistic_scores,"coefficients":media_logistic_coefficients,"uncertainty":media_logistic_uncertainty})

    return (
        media_calibration_curve,
        media_logistic_coefficients,
        media_logistic_folds,
        media_logistic_oof,
        media_logistic_scores,
        media_logistic_uncertainty,
    )


@app.cell(hide_code=True)
def media_failure_logistic_view(
    go,
    make_subplots,
    media_calibration_curve,
    media_logistic_coefficients,
    media_logistic_folds,
    media_logistic_oof,
    media_logistic_scores,
    media_logistic_uncertainty,
    media_tsne_labels,
    mo,
):
    _fig=make_subplots(rows=1,cols=2,subplot_titles=("Held-out-world predicted failure risk","Held-out-world calibration"),horizontal_spacing=.15)
    for _failed,_label,_color in ((False,"Actual pass","#0072B2"),(True,"Actual fail","#D55E00")):
        _fig.add_trace(go.Histogram(x=media_logistic_oof[media_tsne_labels==_failed],name=_label,marker_color=_color,opacity=.6,xbins={"start":0,"end":1,"size":.1},histnorm="probability"),row=1,col=1)
    _fraction,_predicted=media_calibration_curve(media_tsne_labels,media_logistic_oof,n_bins=5,strategy="quantile")
    _fig.add_trace(go.Scatter(x=_predicted,y=_fraction,mode="lines+markers",name="Calibration"),row=1,col=2)
    _fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",line={"dash":"dash","color":"gray"},showlegend=False),row=1,col=2)
    _fig.update_xaxes(title_text="Predicted failure probability",range=[0,1])
    _fig.update_yaxes(title_text="Fraction within outcome class",row=1,col=1)
    _fig.update_yaxes(title_text="Observed failure fraction",range=[0,1],row=1,col=2)
    _fig.update_layout(height=450,template="plotly_white",barmode="overlay",legend={"orientation":"h","y":-.2,"yanchor":"top"},margin={"b":100})
    mo.vstack([mo.md("### Logistic regression: can generation-time features screen channel difficulty?\nTarget = failure of the 10% / 1% individual-media gate in the time-varying-baseline MCMC study. Same ten features as the t-SNE, **not the t-SNE coordinates**. Five folds hold out entire worlds; scaling is fitted only on training worlds. Prespecified L2 regularization (C=1), no hyperparameter tuning. More-media worlds contribute more channel observations.\nThis is an exploratory oracle-feature screen, not a claim about observable real-world inputs or causal drivers. The correlated/deterministically related features make individual coefficient attribution fragile. Coefficients below come from a final full-data fit; predictive metrics use only held-out-world predictions."),mo.ui.plotly(_fig),mo.ui.table(media_logistic_scores,selection=None),mo.ui.table(media_logistic_coefficients,selection=None),mo.ui.table(media_logistic_folds,selection=None),mo.md(str(media_logistic_uncertainty))])

    return


@app.cell(hide_code=True)
def media_smoothing_comparison_functions(
    StudyPath,
    constant_baseline_cfg,
    mo,
    np,
    pg,
    pytensor,
    replace,
    study_asdict,
    study_hashlib,
    study_json,
    study_json_safe,
    tv_thread_limits,
    write_study_json,
):
    from copy import deepcopy as smoothing_deepcopy
    from unittest.mock import patch as smoothing_patch
    from scipy.stats import beta as smoothing_beta_distribution
    from pymc.pytensorf import collect_default_updates as smoothing_collect_rngs
    import pymc_generator.world_model as smoothing_world_model


    def smoothing_capture_batch(batch_id):
        """Replay the unmodified sampler; capture a joint RV snapshot without changing it."""
        cfg=replace(constant_baseline_cfg,n_cells=2,draws_per_cell=1,seed=10029+batch_id)
        built=[];snapshots=[]
        original_build=smoothing_world_model.build_world_model
        original_draw=smoothing_world_model.draw_worlds
        def record_build(g_active,cfg,structural,n_time_steps,prior_cond=None):
            result=original_build(g_active,cfg,structural,n_time_steps,prior_cond=prior_cond)
            built.append({"model":result[0],"outputs":result[1],"g":smoothing_deepcopy(g_active),"structural":smoothing_deepcopy(structural),"prior_cond":smoothing_deepcopy(prior_cond)})
            return result
        def record_draw(model,out_names,seed,draws=1,mode="FAST_COMPILE",*,rng_reference_names=None):
            # The production call is untouched, including its output/RNG traversal.
            ordinary=original_draw(model,out_names,seed,draws=draws,mode=mode,rng_reference_names=rng_reference_names)
            old_vars=[model[n] for n in out_names]
            old_rngs=list(smoothing_collect_rngs(inputs=[],outputs=old_vars))
            if rng_reference_names:
                refs=list(smoothing_collect_rngs(inputs=[],outputs=[model[n] for n in rng_reference_names]))
                old_rngs=refs+[r for r in old_rngs if r not in refs]
            names=tuple(dict.fromkeys((*out_names,*(v.name for v in model.free_RVs))))
            fn,expanded_rngs=smoothing_world_model._get_cached_draw_fn(model,[model[n] for n in names],mode=mode,reference_vars=None)
            # Additional dormant RVs get appended streams; all original RNG ranks stay fixed.
            ordered=old_rngs+[r for r in expanded_rngs if r not in old_rngs]
            values=smoothing_world_model._execute_draws(fn,ordered,seed=seed,draws=draws)
            expanded={n:np.asarray(v)[None] if draws==1 else np.asarray(v) for n,v in zip(names,values)}
            for name in out_names:np.testing.assert_array_equal(expanded[name],ordinary[name],err_msg=f"RNG capture changed {name}")
            snapshots.append({"model":model,"seed":seed,"draws":expanded})
            return ordinary
        # Scoped observers only; no source edits or changed generation decisions.
        with smoothing_patch.object(smoothing_world_model,"build_world_model",record_build),smoothing_patch.object(smoothing_world_model,"draw_worlds",record_draw),tv_thread_limits(limits=1):
            replay=pg.sample_prior_predictive(cfg)
        saved=pg.load_corpus(StudyPath("notebooks/ols_study_50_seed10029")/f"batch_{batch_id:02d}.npz")
        checked=[]
        for key,value in saved.items():
            if isinstance(value,np.ndarray):
                np.testing.assert_array_equal(replay[key],value,err_msg=f"Original corpus replay changed {key}")
                checked.append(key)
        assert len(built)==2
        return cfg,built,snapshots,saved,checked


    def smoothing_generate_batch(batch_id,directory):
        """Evaluate changed smoothing with every exogenous RV held at its original draw."""
        cfg,built,snapshots,saved,checked=smoothing_capture_batch(batch_id)
        reports=[]
        for local,context in enumerate(built):
            w=batch_id*2+local;n_media=int(saved["n_treatments_active"][local])
            matching=[]
            for snapshot in snapshots:
                if snapshot["model"] is context["model"]:
                    for candidate in range(len(snapshot["draws"]["treatments"])):
                        if np.array_equal(snapshot["draws"]["treatments"][candidate].astype(np.float32),saved["treatment_raw"][local,:,:n_media]):
                            matching.append((snapshot,candidate))
            assert len(matching)==1,f"World {w}: expected one exact accepted candidate, found {len(matching)}"
            snapshot,candidate=matching[0]
            original={k:np.asarray(v[candidate]).copy() for k,v in snapshot["draws"].items()}
            structure=smoothing_deepcopy(context["structural"])
            old_s=np.asarray(structure["smoothness_c"])
            new_s=smoothing_beta_distribution.ppf(smoothing_beta_distribution.cdf(old_s,2,2),2,4)
            structure["smoothness_c"]=new_s
            old_width=np.maximum(1,np.rint(26*old_s)).astype(int)
            new_width=np.maximum(1,np.rint(26*new_s)).astype(int)
            assert np.all(new_width<=old_width)
            for key in structure:
                if key!="smoothness_c":np.testing.assert_array_equal(structure[key],context["structural"][key])
            with tv_thread_limits(limits=1):
                model,outputs,_=smoothing_world_model.build_world_model(context["g"],cfg,structure,104,prior_cond=context["prior_cond"])
                assert {v.name for v in model.free_RVs}=={v.name for v in context["model"].free_RVs}
                names=tuple(n for n in outputs if n in original)
                evaluate=pytensor.function(model.free_RVs,[model[n] for n in names],mode="FAST_COMPILE",on_unused_input="ignore")
                new=dict(zip(names,evaluate(*[original[v.name] for v in model.free_RVs])))
            # These graph outputs must be identical at full precision, not just close.
            unchanged=[]
            for key in ("covariates","covariate_contribution","baseline_intrinsic","outcome_noise","saturation_scale","latent_unobserved","confounding_strength"):
                np.testing.assert_array_equal(new[key],original[key],err_msg=f"Smoothing changed protected output {key}")
                unchanged.append(key)
            original_model=context["model"]
            # Independent unchanged-width evaluation verifies the conditioned-RV path too.
            with tv_thread_limits(limits=1):
                replay_fn=pytensor.function(original_model.free_RVs,[original_model[n] for n in names],mode="FAST_COMPILE",on_unused_input="ignore")
                replay_values=replay_fn(*[original[v.name] for v in original_model.free_RVs])
            for name,value in zip(names,replay_values):np.testing.assert_array_equal(value,original[name],err_msg=f"Fixed-input replay changed {name}")
            with np.load(StudyPath("notebooks/time_varying_baseline")/f"world_{w:02d}_data.npz") as tv:
                baseline=tv["baseline"].copy();saved_observed=tv["observed"].copy()
            # Original stored outcomes are left untouched. A paired contribution delta
            # retains exactly the original baseline/noise and float32 storage residual.
            delta=np.sum(new["contributions"]-original["contributions"],axis=1)
            arrays={"media_original":original["treatments"],"media_less_smooth":new["treatments"],
                "contribution_original":original["contributions"],"contribution_less_smooth":new["contributions"],
                "observed_original":saved_observed,"observed_less_smooth":saved_observed+delta,
                "baseline":baseline,"noise":saved["outcome_noise"][local].copy(),"controls":saved["covariates"][local].copy(),
                "control_contributions":saved["covariate_contribution"][local].copy(),
                "smoothness_original":old_s,"smoothness_less_smooth":new_s,"width_original":old_width,"width_less_smooth":new_width,
                "beta":saved["beta"][local,:n_media].copy(),"alpha":saved["carryover_alpha"][local,:n_media].copy(),
                "half":(saved["mm_kappa_mult"][local,:n_media].astype(float)*saved["mechanism_saturation_scale"][local,:n_media].astype(float)),
                "nonzero_effect":np.asarray(context["g"]["g_cy"],dtype=bool)}
            for name in ("media_original","media_less_smooth","contribution_original","contribution_less_smooth"):
                assert np.isfinite(arrays[name]).all() and np.all(arrays[name]>=0)
            assert np.all(arrays["contribution_less_smooth"][:,~arrays["nonzero_effect"]]==0)
            np.testing.assert_allclose(arrays["observed_less_smooth"]-arrays["observed_original"],delta,rtol=1e-10,atol=1e-12)
            np.savez_compressed(directory/f"world_{w:02d}_paired.npz",**arrays)
            np.savez_compressed(directory/f"world_{w:02d}_fixed_rvs.npz",**{v.name:original[v.name] for v in original_model.free_RVs})
            report={"world_id":w,"source_batch":batch_id,"source_seed":cfg.seed,"draw_seed":snapshot["seed"],"accepted_candidate":candidate,
                "active_media":n_media,"nonzero_media":int(arrays["nonzero_effect"].sum()),"source_array_checks":checked,"protected_output_checks":unchanged,
                "fixed_rv_replay_exact":True,"source_array_replay_exact":True,"rng_capture_exact":True,
                "structure_original":study_json_safe(context["structural"]),"structure_less_smooth":study_json_safe(structure),
                "negative_sales_weeks":int(np.sum(arrays["observed_less_smooth"]<0))}
            write_study_json(directory/f"world_{w:02d}_report.json",report);reports.append(report)
        return reports


    def run_media_smoothing_comparison(n_batches=25):
        directory=StudyPath("notebooks/media_smoothing_comparison");directory.mkdir(exist_ok=True)
        sources=list(StudyPath("notebooks/ols_study_50_seed10029").glob("batch_*.npz"))+list(StudyPath("notebooks/time_varying_baseline").glob("world_*_data.npz"))
        hashes={str(p):study_hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}
        manifest={"source_hashes":hashes,"source_config":study_asdict(constant_baseline_cfg),"media_smoothness_old":[2,2],"media_smoothness_new":[2,4],"max_weeks":26,
            "coupling":"Beta(2,4).ppf(Beta(2,2).cdf(original_s)); all primitive RVs fixed",
            "selection":"Original accepted worlds, no reselection under new smoothing; not a fresh accepted-prior sample",
            "baseline":"Preserved matched time-varying baseline","inference":"Not run"}
        manifest=study_json_safe(manifest)
        if (directory/"manifest.json").exists():assert study_json.loads((directory/"manifest.json").read_text())==manifest
        else:write_study_json(directory/"manifest.json",manifest)
        reports=[]
        with mo.status.progress_bar(total=n_batches,title="Matched media-only smoothing comparison") as progress:
            for batch in range(n_batches):
                paths=[directory/f"world_{batch*2+j:02d}_report.json" for j in range(2)]
                if all(p.exists() and (directory/f"world_{batch*2+j:02d}_paired.npz").exists() and (directory/f"world_{batch*2+j:02d}_fixed_rvs.npz").exists() for j,p in enumerate(paths)):
                    reports.extend(study_json.loads(p.read_text()) for p in paths)
                else:reports.extend(smoothing_generate_batch(batch,directory))
                write_study_json(directory/"progress.json",{"worlds_completed":len(reports),"requested_worlds":2*n_batches})
                progress.update()
        for p,h in hashes.items():assert study_hashlib.sha256(StudyPath(p).read_bytes()).hexdigest()==h
        write_study_json(directory/"reports.json",reports)
        return reports


    return (run_media_smoothing_comparison,)


@app.cell(hide_code=True)
def media_smoothing_comparison_run(run_media_smoothing_comparison):
    media_smoothing_reports=run_media_smoothing_comparison(n_batches=25)
    return (media_smoothing_reports,)


@app.cell(hide_code=True)
def media_smoothing_summary_view(
    StudyPath,
    go,
    make_subplots,
    media_smoothing_reports,
    mo,
    np,
    tv_bases,
    write_study_json,
):
    def media_smoothing_walks(world):
        directory=StudyPath("notebooks/media_smoothing_comparison")
        with np.load(directory/f"world_{world:02d}_paired.npz") as a,np.load(directory/f"world_{world:02d}_fixed_rvs.npz") as rv:
            return {label:np.column_stack([np.logaddexp(0,rv["rw_c_mean"][j]+rv["rw_c_std"][j]*(tv_bases[int(width)-1]@rv["eps_c"][:,j]))[8:] for j,width in enumerate(a[f"width_{label}"])]) for label in ("original","less_smooth")}


    def summarize_media_smoothing():
        rows=[]
        for report in media_smoothing_reports:
            w=report["world_id"];walks=media_smoothing_walks(w)
            with np.load(StudyPath("notebooks/media_smoothing_comparison")/f"world_{w:02d}_paired.npz") as a:
                noise_sd=float(a["noise"][7:].std())
                for j,nonzero in enumerate(a["nonzero_effect"]):
                    row={"World":w+1,"Channel":f"C{j+1}","Nonzero effect":bool(nonzero)}
                    for label in ("original","less_smooth"):
                        spend=a[f"media_{label}"][7:,j];contribution=a[f"contribution_{label}"][7:,j];walk=walks[label][7:,j]
                        row.update({f"Width ({label})":int(a[f"width_{label}"][j]),f"Smoothness ({label})":float(a[f"smoothness_{label}"][j]),
                            f"Walk roughness ({label})":float(walk.std() and np.diff(walk).std()/(np.sqrt(2)*walk.std())),
                            f"Spend CV ({label})":float(spend.std()/spend.mean()),
                            f"Spend roughness ({label})":float(np.diff(spend).std()/(np.sqrt(2)*spend.std())) if spend.std() else None,
                            f"Weekly spend change SD / mean ({label})":float(np.diff(spend).std()/spend.mean()),
                            f"Contribution CV ({label})":float(contribution.std()/abs(contribution.mean())) if nonzero and contribution.mean() else None,
                            f"Contribution SD / noise SD ({label})":float(contribution.std()/noise_sd) if nonzero else None,
                            f"Weekly contribution change SD / noise SD ({label})":float(np.diff(contribution).std()/noise_sd) if nonzero else None,
                            f"Mean contribution ({label})":float(contribution.mean()) if nonzero else None})
                    rows.append(row)
        contributing=[r for r in rows if r["Nonzero effect"]]
        summary=[]
        for metric in ("Width","Walk roughness","Spend CV","Spend roughness","Weekly spend change SD / mean","Contribution CV","Contribution SD / noise SD","Weekly contribution change SD / noise SD","Mean contribution"):
            old=np.array([r[f"{metric} (original)"] for r in contributing]);new=np.array([r[f"{metric} (less_smooth)"] for r in contributing]);ratio=new/old
            summary.append({"Quantity":metric,"Original median":float(np.median(old)),"Less-smooth median":float(np.median(new)),"Median paired ratio":float(np.median(ratio)),
                "Paired ratio Q1":float(np.quantile(ratio,.25)),"Paired ratio Q3":float(np.quantile(ratio,.75)),"Increased channels":int(np.sum(new>old+1e-10)),"Decreased channels":int(np.sum(new<old-1e-10)),"Channels":len(old)})
        near_flat={label:sum(r[f"Contribution CV ({label})"]<.1 for r in contributing) for label in ("original","less_smooth")}
        write_study_json(StudyPath("notebooks/media_smoothing_comparison/channel_metrics.json"),rows)
        write_study_json(StudyPath("notebooks/media_smoothing_comparison/summary.json"),{"summary":summary,"near_flat_channels":near_flat,"active_media":len(rows),"contributing_media":len(contributing),"worlds":len(media_smoothing_reports),"recovery":"Not evaluated"})
        return rows,summary,near_flat

    media_smoothing_channel_metrics,media_smoothing_summary,media_smoothing_near_flat=summarize_media_smoothing()
    _fig=make_subplots(rows=2,cols=3,subplot_titles=("Smoothing width (weeks)","Underlying positive-walk roughness","Final spend roughness","Final spend CV","Contribution CV","Contribution SD / noise SD"),vertical_spacing=.2,horizontal_spacing=.12)
    for _i,_metric in enumerate(("Width","Walk roughness","Spend roughness","Spend CV","Contribution CV","Contribution SD / noise SD")):
        _rows=[r for r in media_smoothing_channel_metrics if r["Nonzero effect"]]
        _x=[r[f"{_metric} (original)"] for r in _rows];_y=[r[f"{_metric} (less_smooth)"] for r in _rows]
        _fig.add_trace(go.Scatter(x=_x,y=_y,mode="markers",marker={"size":6,"opacity":.65,"color":"#0072B2"},showlegend=False,text=[f"World {r['World']} · {r['Channel']}" for r in _rows],hovertemplate="%{text}<br>Original %{x:.3f}<br>Less smooth %{y:.3f}<extra></extra>"),row=_i//3+1,col=_i%3+1)
        _lo=min(*_x,*_y);_hi=max(*_x,*_y)
        _fig.add_trace(go.Scatter(x=[_lo,_hi],y=[_lo,_hi],mode="lines",line={"color":"gray","dash":"dash"},showlegend=False),row=_i//3+1,col=_i%3+1)
        _fig.update_xaxes(title_text="Original",row=_i//3+1,col=_i%3+1);_fig.update_yaxes(title_text="Less smooth",row=_i//3+1,col=_i%3+1)
    _fig.update_layout(height=730,template="plotly_white",title="Media-only Beta(2,2) → Beta(2,4): paired generation comparison")
    mo.vstack([mo.md("### Media-only smoothing sensitivity\n**50 matched worlds; no new fits.** Only media smoothness changes from Beta(2,2) to Beta(2,4), using the same prior quantile per channel; the maximum stays 26 weeks. All primitive RV draws—including walk innovations, weekly texture and pulse draws—are held fixed. Controls, mechanism parameters, time-varying baseline and observation noise are preserved.\nOriginal corpus replay and fixed-RV replay were verified exactly. These are counterfactuals of the original accepted worlds, **not a fresh acceptance-filtered sample**. No worlds are dropped or reselected. Statistics use weeks 7–103 and nonzero media effects; all active media were modified. The library recalibrates the walk's expected variance when width changes, so shorter smoothing need not increase total CV. Roughness is SD(weekly differences) / (√2 × temporal SD).\nThe original stored outcome receives the paired media-contribution delta; its original float32 storage residual is retained. No recovery or empirical-realism improvement is claimed."),mo.ui.plotly(_fig),mo.ui.table(media_smoothing_summary,selection=None),mo.md(f"Near-flat contributing channels (CV <10%): **{media_smoothing_near_flat['original']} → {media_smoothing_near_flat['less_smooth']} / 236**."),mo.ui.table(media_smoothing_channel_metrics,selection=None)])

    return media_smoothing_channel_metrics, media_smoothing_walks


@app.cell(hide_code=True)
def media_smoothing_world_picker(media_smoothing_reports, mo):
    media_smoothing_world=mo.ui.dropdown(options={f"World {r['world_id']+1}":r['world_id'] for r in media_smoothing_reports},value="World 1",label="Matched world")
    media_smoothing_world

    return (media_smoothing_world,)


@app.cell(hide_code=True)
def media_smoothing_channel_picker(
    media_smoothing_channel_metrics,
    media_smoothing_world,
    mo,
):
    _rows=[r for r in media_smoothing_channel_metrics if r["World"]==media_smoothing_world.value+1 and r["Nonzero effect"]]
    media_smoothing_channel=mo.ui.dropdown(options={r["Channel"]:int(r["Channel"][1:])-1 for r in _rows},value=_rows[0]["Channel"],label="Contributing media channel")
    media_smoothing_channel

    return (media_smoothing_channel,)


@app.cell(hide_code=True)
def media_smoothing_trajectory_comparison(
    StudyPath,
    go,
    make_subplots,
    media_smoothing_channel,
    media_smoothing_walks,
    media_smoothing_world,
    mo,
    np,
):
    _w=media_smoothing_world.value;_j=media_smoothing_channel.value
    _walks=media_smoothing_walks(_w)
    with np.load(StudyPath("notebooks/media_smoothing_comparison")/f"world_{_w:02d}_paired.npz") as _a:
        _old=int(_a["width_original"][_j]);_new=int(_a["width_less_smooth"][_j])
        _fig=make_subplots(rows=3,cols=1,subplot_titles=("Underlying positive walk, before added texture and schedules","Final raw spend (before adstock)","True contribution (after adstock and saturation)"),vertical_spacing=.13)
        for _label,_name,_color in (("original",f"Original: {_old}-week window","#0072B2"),("less_smooth",f"Less smooth: {_new}-week window","#D55E00")):
            for _row,_values in ((1,_walks[_label][:,_j]),(2,_a[f"media_{_label}"][:,_j]),(3,_a[f"contribution_{_label}"][:,_j])):
                _fig.add_trace(go.Scatter(x=np.arange(104),y=_values,mode="lines",name=_name,legendgroup=_label,showlegend=_row==1,line={"color":_color,"width":2,"dash":"solid" if _label=="original" else "dash"}),row=_row,col=1)
        _fig.add_vrect(x0=-.5,x1=6.5,fillcolor="gray",opacity=.1,line_width=0,row="all",col=1)
        _fig.update_xaxes(title_text="Reported week");_fig.update_layout(height=870,template="plotly_white",title=f"World {_w+1} · C{_j+1} · same innovations and mechanisms",legend={"orientation":"h","y":-.09,"yanchor":"top"},margin={"b":100})
    mo.vstack([mo.ui.plotly(_fig),mo.md("Shading marks weeks 0–6, excluded from the summary metrics. These are **true generated trajectories**, not fitted contributions. Overlap in final spend can be substantial because its high-frequency texture and schedules are unchanged.")])

    return


@app.cell(hide_code=True)
def media_smoothing_final_validation(
    StudyPath,
    media_smoothing_channel_metrics,
    media_smoothing_reports,
    mo,
    np,
    study_hashlib,
    study_json,
    transform_media_columns,
    write_study_json,
):
    def validate_media_smoothing_comparison():
        rows=[r for r in media_smoothing_channel_metrics if r['Nonzero effect']]
        max_transform_error=0.0;unchanged_width_channels=0
        for report in media_smoothing_reports:
         w=report['world_id'];path=StudyPath('notebooks/media_smoothing_comparison')/f'world_{w:02d}_paired.npz'
         with np.load(path) as a:
          old=a['media_original'];new=a['media_less_smooth'];mask=a['nonzero_effect'];unchanged=a['width_original']==a['width_less_smooth']
          np.testing.assert_array_equal(old[:,unchanged],new[:,unchanged]);unchanged_width_channels+=int(unchanged.sum())
          np.testing.assert_array_equal(a['contribution_original'][:,unchanged],a['contribution_less_smooth'][:,unchanged])
          for label in ['original','less_smooth']:
           computed=transform_media_columns(a[f'media_{label}'],a['alpha'],a['half'])[7:]*a['beta']*mask
           error=float(np.max(np.abs(computed-a[f'contribution_{label}'][7:])))
           max_transform_error=max(max_transform_error,error)
           np.testing.assert_allclose(computed,a[f'contribution_{label}'][7:],rtol=3e-6,atol=3e-6)
          with np.load(f'notebooks/time_varying_baseline/world_{w:02d}_data.npz') as source:
           np.testing.assert_array_equal(a['baseline'],source['baseline']);np.testing.assert_array_equal(a['observed_original'],source['observed'])
          np.testing.assert_allclose(a['observed_less_smooth']-a['observed_original'],(a['contribution_less_smooth']-a['contribution_original']).sum(axis=1),rtol=1e-10,atol=1e-12)
        manifest=study_json.loads(StudyPath('notebooks/media_smoothing_comparison/manifest.json').read_text())
        for path,digest in manifest['source_hashes'].items():assert study_hashlib.sha256(StudyPath(path).read_bytes()).hexdigest()==digest
        validation={'worlds':len(media_smoothing_reports),'active_media':len(media_smoothing_channel_metrics),'contributing_media':len(rows),'source_hashes_unchanged':len(manifest['source_hashes']),
         'source_corpus_array_replay_exact':all(r['source_array_replay_exact'] for r in media_smoothing_reports),
         'fixed_rv_replay_exact':all(r['fixed_rv_replay_exact'] for r in media_smoothing_reports),
         'unchanged_width_active_channels_exact':unchanged_width_channels,'maximum_independent_transform_difference_with_float32_parameters':max_transform_error,
         'baseline_unchanged':True,'observation_delta_only_media':True,'negative_sales_weeks':sum(r['negative_sales_weeks'] for r in media_smoothing_reports),
         'inference_rerun':False}
        write_study_json(StudyPath('notebooks/media_smoothing_comparison/validation.json'),validation)
        return validation

    media_smoothing_validation=validate_media_smoothing_comparison()
    mo.md("**Media-only smoothing validation:** " + str(media_smoothing_validation))
    return


if __name__ == "__main__":
    app.run()
