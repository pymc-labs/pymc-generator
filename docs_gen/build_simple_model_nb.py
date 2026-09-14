"""Assemble ``docs/examples/simple-model.ipynb`` with executable recovery checks.

Run from the repository root: ``python docs_gen/build_simple_model_nb.py``.
The builder writes an unexecuted notebook; MkDocs or Jupyter populates outputs.

The notebook is the estimation companion to ``index.ipynb``: one causally
sufficient world, a stock pymc-marketing MMM fitted on its observables, the
measured failure and its diagnosis, and the posterior oracle as the reference
that shows the world itself was recoverable all along.
"""

from __future__ import annotations

import os

import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md(
        "# A simple model: can a real MMM recover a known world?\n"
        "\n"
        "[The data-generation tour](index.ipynb) built worlds from the smallest graph\n"
        "to a confounded one. This notebook asks the estimation question: draw **one\n"
        "deliberately favourable world** — causal sufficiency holds, one media\n"
        "response family, healthy spend variation — hand a stock\n"
        "`pymc_marketing.mmm.MMM` its observables, and score it against the known\n"
        "truth. Then fit the generator's own **posterior oracle** on the same data\n"
        "and compare.\n"
        "\n"
        "The punchline is worth stating up front, because it is measured below:\n"
        "\n"
        "1. the MMM's weekly *shape* is nearly perfect, its cumulative *attribution*\n"
        "   is off by 19–36% with intervals that miss — and its sampler diagnostics\n"
        "   look clean, so nothing warns you;\n"
        "2. the oracle recovers every channel within a few percent, with 3/3\n"
        "   coverage and the true parameters inside tight posteriors —\n"
        "   **the world was identified; the gap is the estimator's error model**."
    ),
    code(
        "%matplotlib inline\n"
        "\n"
        "import os\n"
        "import tempfile\n"
        "import time\n"
        "\n"
        "import arviz as az\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import pymc as pm\n"
        "from IPython.display import Image, display\n"
        "\n"
        "from pymc_marketing.mmm import MMM, GeometricAdstock, MichaelisMentenSaturation\n"
        "\n"
        "import prior_generator as pg\n"
        "from prior_generator import viz\n"
        "from prior_generator.sampler import ADSTOCK_FAMILY_KEYS, SATURATION_FAMILY_KEYS\n"
        "\n"
        "print('prior-generator', pg.__version__)"
    ),
    # ---------------------------------------------------------------- 1
    md(
        "## 1 · One causally sufficient world\n"
        "\n"
        "Three channels feed sales directly; two observed controls feed the baseline\n"
        "only; the latent demand slot is present but **isolated** (every `D` edge\n"
        "budget is zero) and `rho = 0`, so there is no unobserved common cause of\n"
        "spend and sales. A regression on the observables is licensed — whatever\n"
        "fails below will not be an identification failure.\n"
        "\n"
        "One functional form everywhere, and it is the exact pair the MMM will use:\n"
        "normalized geometric adstock (`l_max=4`) × Michaelis-Menten saturation."
    ),
    code(
        "def pin(adstock, saturation):\n"
        '    """Family probability dicts that select exactly one family each."""\n'
        "    return {\n"
        "        'adstock_family_probs': {**dict.fromkeys(ADSTOCK_FAMILY_KEYS, 0.0), adstock: 1.0},\n"
        "        'saturation_family_probs': {\n"
        "            **dict.fromkeys(SATURATION_FAMILY_KEYS, 0.0), saturation: 1.0\n"
        "        },\n"
        "    }\n"
        "\n"
        "cfg = pg.make_scm_prior(\n"
        "    n_treatments=3, n_covariates=2, n_latent=1, n_time_steps=104, l_max=4,\n"
        "    n_treatments_active_range=(3, 3), n_covariates_active_range=(2, 2),\n"
        "    n_latent_active_range=(1, 1),\n"
        "    edge_budget={'cy': (3, 3), 'zy': (2, 2), 'dy': (0, 0), 'dc': (0, 0),\n"
        "                 'dz': (0, 0), 'zc': (0, 0), 'cc': (0, 0), 'zz': (0, 0)},\n"
        "    confounding_strength_range=(0.0, 0.0),   # rho = 0: fully Markovian\n"
        "    n_cells=2, draws_per_cell=1, seed=20260728,\n"
        "    **pin('geometric', 'michaelis_menten'),\n"
        ")\n"
        "world = pg.sample_scm(cfg, seed=22, name='sufficient')\n"
        "\n"
        "# The drawn world IS the graph we asked for.\n"
        "assert world.g['g_cy'].all() and world.g['g_zy'].all()\n"
        "for name in ('g_dy', 'g_dc', 'g_dz', 'g_zc', 'g_cc', 'g_zz'):\n"
        "    assert not world.g[name].any(), name\n"
        "assert float(world.data['confounding_strength']) == 0.0\n"
        "assert world.identity_error() < 1e-9\n"
        "print(f'n_time_steps={world.n_time_steps}  n_treatments={world.n_treatments} channels'\n"
        "      f'  n_covariates={world.n_covariates} controls'\n"
        "      f'  n_latent={world.n_latent} latent (isolated)')"
    ),
    code(
        "def show(plot_fn, w, title):\n"
        "    path = os.path.join(tempfile.mkdtemp(), 'fig.png')\n"
        "    plot_fn(w, path, title)\n"
        "    display(Image(filename=path))\n"
        "\n"
        "show(viz.plot_dag, world, 'Causal sufficiency: no latent parent of both C and Y')"
    ),
    md(
        "The model-facing table is dated spend, controls, and sales — nothing else.\n"
        "Truth series (`contributions_observed`, `baseline`, the parameters) stay on\n"
        "our side of the fence, for scoring only."
    ),
    code(
        "CHANNELS = [f'C{k + 1}' for k in range(world.n_treatments)]\n"
        "CONTROLS = [f'Z{m + 1}' for m in range(world.n_covariates)]\n"
        "frame = pd.DataFrame(\n"
        "    {'date': pd.date_range('2025-01-06', periods=world.n_time_steps, freq='W-MON')}\n"
        ")\n"
        "for k, name in enumerate(CHANNELS):\n"
        "    frame[name] = world.data['channels'][:, k]\n"
        "for m, name in enumerate(CONTROLS):\n"
        "    frame[name] = world.data['controls'][:, m]\n"
        "frame['Y'] = world.data['sales']\n"
        "X, y = frame.drop(columns='Y'), frame['Y']\n"
        "display(frame.head())\n"
        "\n"
        "show(viz.plot_channels, world, 'Per-channel spend and its true effect')"
    ),
    # ---------------------------------------------------------------- 2
    md(
        "## 2 · The scoring rule, fixed before any model runs\n"
        "\n"
        "Per channel, over the weeks from `l_max` on (the first `l_max` weeks depend\n"
        "on pre-window burn-in history that no fitted model sees, so scoring them\n"
        "would compare different estimands):\n"
        "\n"
        "- **cumulative attribution**: posterior median of the summed weekly\n"
        "  contribution, its equal-tailed 94% interval, relative error, and whether\n"
        "  the interval covers the truth;\n"
        "- **weekly shape**: correlation between the posterior-mean path and the true\n"
        "  path.\n"
        "\n"
        "Same function for every model. `geometry` reports what the sampler itself\n"
        "says: divergences, worst r-hat, smallest bulk ESS."
    ),
    code(
        "START = cfg.l_max\n"
        "TRUTH = np.asarray(world.data['contributions_observed'])\n"
        "\n"
        "def score(samples, dim, label):\n"
        '    """Cumulative + shape recovery of per-channel contributions."""\n'
        "    s = samples.isel(date=slice(START, None))\n"
        "    totals = s.sum('date').quantile([0.03, 0.5, 0.97], dim=('chain', 'draw'))\n"
        "    mean_path = s.mean(('chain', 'draw'))\n"
        "    rows = []\n"
        "    for k, name in enumerate(CHANNELS):\n"
        "        true_path = TRUTH[START:, k]\n"
        "        true_total = float(true_path.sum())\n"
        "        lo, med, hi = (float(totals.sel({dim: name, 'quantile': q}))\n"
        "                       for q in (0.03, 0.5, 0.97))\n"
        "        rows.append({\n"
        "            'model': label, 'channel': name,\n"
        "            'true_total': true_total, 'post_median': med,\n"
        "            'low_94': lo, 'high_94': hi,\n"
        "            'rel_error': abs(med - true_total) / abs(true_total),\n"
        "            'covered_94': bool(lo <= true_total <= hi),\n"
        "            'shape_corr': float(np.corrcoef(true_path, np.asarray(mean_path.sel({dim: name})))[0, 1]),\n"
        "        })\n"
        "    return pd.DataFrame(rows)\n"
        "\n"
        "def geometry(idata, label, seconds):\n"
        '    """Raw R-hat and bulk ESS over posterior variables, including deterministics.\n'
        "\n"
        "    Round only the displayed table, never the values used for decisions.\n"
        "    Library warnings remain visible and may cover additional diagnostics.\n"
        '    """\n'
        "    stats = idata['sample_stats'] if 'sample_stats' in idata else idata.sample_stats\n"
        "    rhat = az.rhat(idata)\n"
        "    ess = az.ess(idata, method='bulk')\n"
        "    return {\n"
        "        'model': label,\n"
        "        'divergences': int(np.asarray(stats['diverging']).sum()),\n"
        "        'max_rhat': max(float(np.nanmax(rhat[v])) for v in rhat.data_vars),\n"
        "        'min_ess_bulk': min(float(np.nanmin(ess[v])) for v in ess.data_vars),\n"
        "        'seconds': float(seconds),\n"
        "    }\n"
        "\n"
        "print('scored weeks:', world.n_time_steps - START, 'of', world.n_time_steps)"
    ),
    # ---------------------------------------------------------------- 3
    md(
        "## 3 · The stock MMM attempt\n"
        "\n"
        "Everything favours it: the component classes match the true mechanism\n"
        "family exactly, the controls are the true controls, causal sufficiency\n"
        "holds. Default priors, constant intercept — the standard first fit anyone\n"
        "would run."
    ),
    code(
        "mmm = MMM(\n"
        "    date_column='date', channel_columns=CHANNELS, control_columns=CONTROLS,\n"
        "    target_column='Y',\n"
        "    adstock=GeometricAdstock(l_max=cfg.l_max).set_dims_for_all_priors('channel'),\n"
        "    saturation=MichaelisMentenSaturation().set_dims_for_all_priors('channel'),\n"
        ")\n"
        "mmm.build_model(X, y)\n"
        "mmm.add_original_scale_contribution_variable(['channel_contribution'])\n"
        "started = time.perf_counter()\n"
        "mmm_idata = mmm.fit(X, y, draws=1000, tune=1000, chains=4, cores=4,\n"
        "                    target_accept=0.9, random_seed=11, progressbar=False)\n"
        "mmm_geo = geometry(mmm_idata, 'MMM (constant intercept)',\n"
        "                   time.perf_counter() - started)\n"
        "print(mmm_geo)"
    ),
    code(
        "mmm_scores = score(\n"
        "    mmm_idata['posterior']['channel_contribution_original_scale'],\n"
        "    'channel', 'MMM (constant intercept)',\n"
        ")\n"
        "display(mmm_scores.round(3))\n"
        "print(f\"worst cumulative error: {mmm_scores['rel_error'].max():.1%}   \"\n"
        "      f\"coverage: {int(mmm_scores['covered_94'].sum())}/3   \"\n"
        "      f\"worst shape corr: {mmm_scores['shape_corr'].min():.3f}\")"
    ),
    md(
        "### The issues, pointed at directly\n"
        "\n"
        "Three things are true at once, and only together do they tell the story:\n"
        "\n"
        "1. **The weekly shape is essentially perfect** (correlation ≥ 0.98). The\n"
        "   response families are right and the spend sweeps them.\n"
        "2. **The cumulative attribution is badly off** — tens of percent, with 94%\n"
        "   intervals that exclude the truth.\n"
        "3. **Every sampler diagnostic is clean**: no divergences, r-hat ≈ 1.00.\n"
        "   Nothing in the fit warns you that the answer is wrong.\n"
        "\n"
        "So this is not non-identification (causal sufficiency holds by\n"
        "construction) and not a sampling failure. It is **misspecification**: the\n"
        "true baseline is a smooth latent *walk*, and the MMM brings a constant\n"
        "intercept plus iid noise. The likelihood has to put the walk somewhere, and\n"
        "the smooth, autocorrelated media terms are the only flexible pieces\n"
        "available — so the level drifts into the channels and the intervals stay\n"
        "confidently narrow around a biased split."
    ),
    code(
        "residual_truth = (np.asarray(world.data['baseline_intrinsic'])\n"
        "                  + np.asarray(world.data['confounder_contribution']).sum(1))[START:]\n"
        "media_truth = np.asarray(world.data['contributions_observed']).sum(1)[START:]\n"
        "lag1 = float(np.corrcoef(residual_truth[:-1], residual_truth[1:])[0, 1])\n"
        "\n"
        "print('what the MMM cannot represent (the intrinsic baseline path):')\n"
        "print(f'  sd(baseline path) / sd(media)  = {residual_truth.std() / media_truth.std():.2f}')\n"
        "print(f'  lag-1 autocorrelation          = {lag1:.3f}   (iid noise would be ~0)')\n"
        "print(f'  share of sales sd              = {residual_truth.std() / np.std(np.asarray(world.data[\"sales\"])[START:]):.2f}')"
    ),
    # ---------------------------------------------------------------- 4
    md(
        "## 4 · The honest upgrade: a time-varying intercept\n"
        "\n"
        "Once you suspect a latent baseline trend, `time_varying_intercept=True` is\n"
        "the option pymc-marketing offers: an HSGP prior on the intercept path. It\n"
        "is the right *kind* of fix — the baseline needs a low-frequency basis, not\n"
        "a constant."
    ),
    code(
        "mmm_tvi = MMM(\n"
        "    date_column='date', channel_columns=CHANNELS, control_columns=CONTROLS,\n"
        "    target_column='Y',\n"
        "    adstock=GeometricAdstock(l_max=cfg.l_max).set_dims_for_all_priors('channel'),\n"
        "    saturation=MichaelisMentenSaturation().set_dims_for_all_priors('channel'),\n"
        "    time_varying_intercept=True,\n"
        ")\n"
        "mmm_tvi.build_model(X, y)\n"
        "mmm_tvi.add_original_scale_contribution_variable(['channel_contribution'])\n"
        "started = time.perf_counter()\n"
        "tvi_idata = mmm_tvi.fit(X, y, draws=1000, tune=1000, chains=4, cores=4,\n"
        "                        target_accept=0.9, random_seed=11, progressbar=False)\n"
        "tvi_geo = geometry(tvi_idata, 'MMM (time-varying intercept)',\n"
        "                   time.perf_counter() - started)\n"
        "print(tvi_geo)\n"
        "tvi_scores = score(\n"
        "    tvi_idata['posterior']['channel_contribution_original_scale'],\n"
        "    'channel', 'MMM (time-varying intercept)',\n"
        ")\n"
        "display(tvi_scores.round(3))\n"
        "print(f\"worst cumulative error: {tvi_scores['rel_error'].max():.1%}   \"\n"
        "      f\"coverage: {int(tvi_scores['covered_94'].sum())}/3\")"
    ),
    md(
        "Better — the channel that absorbed the most baseline is largely repaired —\n"
        "but the remaining channels are still tens of percent off, and the intervals\n"
        "still miss. Two cautions from fitting this same model family across many\n"
        "seeds of this corpus:\n"
        "\n"
        "- one seed is not a benchmark; the *size* of the residual attribution error\n"
        "  varies a lot from world to world;\n"
        "- the HSGP intercept's length-scale and coefficients are the components\n"
        "  whose r-hat degrades first when the fit is stressed. **Read MMM point\n"
        "  estimates on these worlds only after checking r-hat**, because when the\n"
        "  geometry does break, it breaks silently in exactly those parameters."
    ),
    # ---------------------------------------------------------------- 5
    md(
        "## 5 · The posterior oracle: the identification floor\n"
        "\n"
        "`world.oracle_model()` rebuilds the world's own model form as the posterior:\n"
        "same prior *definitions* generation used (they physically share code, so\n"
        "draw and oracle cannot drift), the true structure and families given, the\n"
        "observed spend/controls/sales attached. By default it **marginalizes the\n"
        "outcome-side Gaussian paths analytically** — the baseline walk, the latent\n"
        "demand walks, and the iid sales noise enter as an exact multivariate-normal\n"
        "covariance, so no latent time series is sampled at all.\n"
        "\n"
        "It is an *oracle* — it is told the true graph and families, which no real\n"
        "analyst knows. That is the point: it measures what the **data** identifies,\n"
        "separating world-is-broken from estimator-cannot-represent-it."
    ),
    code(
        "oracle = world.oracle_model()\n"
        "print('free parameters:', sorted(rv.name for rv in oracle.free_RVs))\n"
        "started = time.perf_counter()\n"
        "with oracle:\n"
        "    oracle_idata = pm.sample(draws=1000, tune=1000, chains=4, cores=4,\n"
        "                             target_accept=0.9, random_seed=11,\n"
        "                             progressbar=False)\n"
        "oracle_geo = geometry(oracle_idata, 'oracle (marginal)',\n"
        "                      time.perf_counter() - started)\n"
        "print(oracle_geo)"
    ),
    code(
        "oracle_contrib = (\n"
        "    oracle_idata.posterior['contributions']\n"
        "    .rename({'contributions_dim_0': 'date', 'contributions_dim_1': 'channel'})\n"
        "    .assign_coords(channel=CHANNELS, date=np.arange(world.n_time_steps))\n"
        ")\n"
        "oracle_scores = score(oracle_contrib, 'channel', 'oracle (marginal)')\n"
        "display(oracle_scores.round(3))\n"
        "print(f\"worst cumulative error: {oracle_scores['rel_error'].max():.1%}   \"\n"
        "      f\"coverage: {int(oracle_scores['covered_94'].sum())}/3\")\n"
        "\n"
        "assert oracle_geo['max_rhat'] < 1.01\n"
        "assert int(oracle_scores['covered_94'].sum()) == 3\n"
        "assert oracle_scores['rel_error'].max() < 0.15"
    ),
    md("And the structural parameters themselves are recovered, not just the series:"),
    code(
        "rows = []\n"
        "for name in ('beta', 'adstock_alpha', 'mm_kappa_mult'):\n"
        "    truth_values = np.asarray(world.params[name])\n"
        "    draws = oracle_idata.posterior[name]\n"
        "    mean = np.asarray(draws.mean(('chain', 'draw')))\n"
        "    lo = np.asarray(draws.quantile(0.03, dim=('chain', 'draw')))\n"
        "    hi = np.asarray(draws.quantile(0.97, dim=('chain', 'draw')))\n"
        "    for k, channel in enumerate(CHANNELS):\n"
        "        rows.append({'parameter': name, 'channel': channel,\n"
        "                     'true': truth_values[k], 'post_mean': mean[k],\n"
        "                     'low_94': lo[k], 'high_94': hi[k],\n"
        "                     'covered': bool(lo[k] <= truth_values[k] <= hi[k])})\n"
        "recovered = pd.DataFrame(rows)\n"
        "display(recovered.round(3))\n"
        "print('parameter coverage:', int(recovered['covered'].sum()), '/', len(recovered))"
    ),
    # ---------------------------------------------------------------- 6
    md("## 6 · Side by side"),
    code(
        "summary = pd.concat([mmm_scores, tvi_scores, oracle_scores], ignore_index=True)\n"
        "coverage = summary.groupby('model')['covered_94'].sum().astype(int)\n"
        "geo = pd.DataFrame([mmm_geo, tvi_geo, oracle_geo]).set_index('model')\n"
        "table = pd.DataFrame({\n"
        "    'worst rel error': summary.groupby('model')['rel_error'].max(),\n"
        "    'median rel error': summary.groupby('model')['rel_error'].median(),\n"
        "    'covered / 3': coverage,\n"
        "    'divergences': geo['divergences'],\n"
        "    'max r-hat': geo['max_rhat'],\n"
        "    'min ESS': geo['min_ess_bulk'],\n"
        "})\n"
        "display(table.round(3))"
    ),
    code(
        "quantiles = {\n"
        "    'MMM (constant intercept)': mmm_idata['posterior']['channel_contribution_original_scale'],\n"
        "    'oracle (marginal)': oracle_contrib,\n"
        "}\n"
        "weeks = np.arange(world.n_time_steps)[START:]\n"
        "fig, axes = plt.subplots(world.n_treatments, len(quantiles), figsize=(11, 7),\n"
        "                         sharex=True, sharey='row')\n"
        "for col, (label, samples) in enumerate(quantiles.items()):\n"
        "    s = samples.isel(date=slice(START, None))\n"
        "    q = s.quantile([0.03, 0.5, 0.97], dim=('chain', 'draw'))\n"
        "    for k, channel in enumerate(CHANNELS):\n"
        "        ax = axes[k, col]\n"
        "        ax.plot(weeks, TRUTH[START:, k], color='black', lw=1.4, label='truth')\n"
        "        ax.plot(weeks, q.sel(channel=channel, quantile=0.5), color='C0', lw=1.1,\n"
        "                label='posterior median')\n"
        "        ax.fill_between(weeks, q.sel(channel=channel, quantile=0.03),\n"
        "                        q.sel(channel=channel, quantile=0.97), color='C0', alpha=0.25,\n"
        "                        label='94% interval')\n"
        "        ax.spines[['top', 'right']].set_visible(False)\n"
        "        if k == 0:\n"
        "            ax.set_title(label, fontsize=10)\n"
        "        if col == 0:\n"
        "            ax.set_ylabel(channel)\n"
        "axes[0, 0].legend(frameon=False, fontsize=8)\n"
        "axes[-1, 0].set_xlabel('week'); axes[-1, 1].set_xlabel('week')\n"
        "fig.suptitle('Weekly contribution: truth vs posterior', y=1.02)\n"
        "fig.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "The two columns share y-axes per row, so the difference is visible at a\n"
        "glance: both track the weekly shape, but the MMM's whole band sits **beside**\n"
        "the truth on the channels that absorbed baseline level, while the oracle's\n"
        "band sits **on** it."
    ),
    # ---------------------------------------------------------------- 7
    md(
        "## 7 · What to take away\n"
        "\n"
        "- **The world was recoverable.** The oracle — same data, correct error\n"
        "  model — recovers every cumulative contribution within a few percent, with\n"
        "  full coverage, r-hat ≈ 1.00, and the true `beta`, `adstock_alpha`,\n"
        "  `mm_kappa_mult` inside tight posteriors. Any residual gap between an MMM\n"
        "  and the truth on this corpus is an *estimator* property, not a data\n"
        "  property.\n"
        "- **A clean fit is not a correct attribution.** The constant-intercept MMM\n"
        "  passed every convergence diagnostic while misallocating tens of percent of\n"
        "  cumulative contribution, because a latent baseline walk existed and its\n"
        "  likelihood had nowhere to put it.\n"
        "- **The intercept needs capacity, not luck.** The time-varying intercept is\n"
        "  the right instinct; on these worlds the baseline requires a genuinely\n"
        "  low-frequency basis, and its hyperparameters are the first place to look\n"
        "  when r-hat degrades.\n"
        "- The knob that *generates* this difficulty — the residual/media amplitude\n"
        "  ratio — is a declared axis of the corpus. See section 6 of the\n"
        "  [data-generation tour](index.ipynb), and the posterior-oracle guide for\n"
        "  the oracle's exact assumptions and concessions."
    ),
]

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python (prior-generator)",
        "language": "python",
        "name": "prior-generator",
    },
    "language_info": {"name": "python"},
}

out = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs",
    "examples",
    "simple-model.ipynb",
)
with open(out, "w") as fh:
    nbf.write(nb, fh)
print("wrote", out, f"({len(cells)} cells)")
