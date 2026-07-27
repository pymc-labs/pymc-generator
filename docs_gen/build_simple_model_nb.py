"""Assemble ``docs/examples/simple-model.ipynb`` with executable corpus checks.

Run from the repository root: ``python docs_gen/build_simple_model_nb.py``.
The builder writes an unexecuted notebook; MkDocs or Jupyter populates outputs.
"""

from __future__ import annotations

import os

import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md(
        "# A simple configurable corpus model\n\n"
        "This executable example builds the smallest useful variable-size corpus: at most "
        "three media channels and two observed controls. Every active channel directly "
        "affects sales. Controls affect sales through the baseline only; they never alter "
        "channel nodes. One latent demand slot is present but deliberately isolated, so this "
        "is not a confounding example.\n\n"
        "The easy and difficult variants use the same graph and mechanism family. They differ "
        "only in the sales random-walk noise prior."
    ),
    code(
        "import os\n"
        "import shutil\n"
        "import tempfile\n"
        "import warnings\n\n"
        "warnings.filterwarnings('ignore', message='IProgress not found')\n\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import pymc as pm\n"
        "from IPython.display import Image, display\n\n"
        "from pymc_marketing.mmm import GeometricAdstock, MichaelisMentenSaturation, MMM\n\n"
        "import prior_generator as pg\n"
        "from prior_generator import viz\n"
        "from prior_generator.data_generator import DataGenerator\n"
        "from prior_generator.slots import SlotLayout\n"
        "from prior_generator.sampler import ADSTOCK_FAMILY_KEYS, SATURATION_FAMILY_KEYS\n"
        "print('prior-generator', pg.__version__)"
    ),
    md(
        "## Fixed maximum layout and exact graph budget\n\n"
        "Active counts are sampled per cell, then inactive trailing slots are zero padded. "
        "All eight edge budgets are explicit, so no omitted Bernoulli default can add an edge. "
        "The maximum exact `zb` budget clamps to the eligible active controls, so each one "
        "affects the baseline. The zero `zc` budget forbids every control-to-channel edge.\n\n"
        "Family configuration uses canonical names, so it is self-describing rather than positional. "
        "The unchanged corpus schema intentionally does not persist `sat_family`; the full single-world "
        "audit below exposes both selected mechanism families."
    ),
    code(
        "EDGE_BUDGET = {\n"
        "    'cy': (3, 3), 'dc': (0, 0), 'dz': (0, 0), 'db': (0, 0),\n"
        "    'zb': (2, 2), 'zc': (0, 0), 'cc': (0, 0), 'zz': (0, 0),\n"
        "}\n"
        "COMMON = dict(\n"
        "    n_treatments=3, n_covariates=2, n_latent=1, T=48,\n"
        "    n_treatments_active_range=(1, 3),\n"
        "    n_covariates_active_range=(1, 2),\n"
        "    n_latent_active_range=(1, 1),\n"
        "    edge_budget=EDGE_BUDGET, n_cells=6, draws_per_cell=1, seed=20260726,\n"
        "    adstock_family_probs={**dict.fromkeys(ADSTOCK_FAMILY_KEYS, 0.0), 'geometric': 1.0},\n"
        "    saturation_family_probs={**dict.fromkeys(SATURATION_FAMILY_KEYS, 0.0), 'michaelis_menten': 1.0},\n"
        "    confounding_strength_range=(0.0, 0.0),\n"
        ")\n"
        "easy_cfg = pg.make_scm_prior(**COMMON, rw_sales_std_sigma=0.05)\n"
        "hard_cfg = pg.make_scm_prior(**COMMON, rw_sales_std_sigma=1.0)\n"
        "assert easy_cfg.adstock_family_probs == {**dict.fromkeys(ADSTOCK_FAMILY_KEYS, 0.0), 'geometric': 1.0}\n"
        "assert easy_cfg.saturation_family_probs == {**dict.fromkeys(SATURATION_FAMILY_KEYS, 0.0), 'michaelis_menten': 1.0}\n"
        "print('layout slots:', easy_cfg.layout.n_slots, '| edge budget:', EDGE_BUDGET)"
    ),
    md(
        "## A full-size visual world\n\n"
        "The corpus below uses `sample_prior_predictive` to vary active counts and pad inactive "
        "slots. For these visualizations only, `sample_scm` draws one world with all configured "
        "maximum-size nodes active: C1 through C3, Z1 through Z2, and the isolated D1 node. "
        "Its graph still has no D1 edges, each control has only a Z to B path, and every channel "
        "has geometric adstock and Michaelis-Menten saturation."
    ),
    code(
        "world = pg.sample_scm(easy_cfg, seed=20260726)\n"
        "assert world.K == 3 and world.M == 2 and world.J == 1\n"
        "assert not world.g['g_dc'].any() and not world.g['g_dz'].any() and not world.g['g_db'].any()\n"
        "assert world.g['g_cy'].all() and world.g['g_zb'].all()\n"
        "assert not world.g['g_zc'].any() and not world.g['g_cc'].any() and not world.g['g_zz'].any()\n"
        "print('visual world:', f'T={world.T}, K={world.K}, M={world.M}, J={world.J}')\n"
        "print('D1 is isolated; Z1/Z2 affect only the baseline, never C1/C2/C3.')"
    ),
    md(
        "## Structural-equation audit\n\n"
        "`sample_scm` records this world as an acyclic Pearl SCM over vector-valued nodes. "
        "The vectors span the full generated horizon; this does not assert a Markov property over "
        "time. `world.equations` gives the executed vector equations, while "
        "`world.equation_parameters` gives their realized coefficients and mechanism choices. "
        "With `rho = 0`, the recorded exogenous vectors are independent. Exact replay requires "
        "both the realized equation parameters and the recorded exogenous innovations."
    ),
    code(
        "equations = world.equations\n"
        "equation_parameters = world.equation_parameters\n"
        "exogenous = world.exogenous\n"
        "T_full = world.T + world.cfg.adstock_burn_in\n"
        "audit_nodes = ('D1', 'Z1', 'Z2', 'C1', 'C2', 'C3', 'B', 'Y')\n"
        "assert set(audit_nodes) <= equations.keys()\n"
        "assert set(audit_nodes) <= equation_parameters.keys()\n"
        "for node in audit_nodes:\n"
        "    print(f'\\n{node}: {equations[node]}')\n"
        "    print('parameters:', equation_parameters[node])\n\n"
        "def family_from_audit(parameters, mechanism):\n"
        "    family_keys = (f'{mechanism}_family', 'sat_family', mechanism) if mechanism == 'saturation' else ('adstock_family', mechanism)\n"
        "    for key in family_keys:\n"
        "        if key in parameters:\n"
        "            value = parameters[key]\n"
        "            return value['family'] if isinstance(value, dict) else value\n"
        "    for value in parameters.values():\n"
        "        if isinstance(value, dict):\n"
        "            try:\n"
        "                return family_from_audit(value, mechanism)\n"
        "            except AssertionError:\n"
        "                pass\n"
        "    raise AssertionError(f'{mechanism} family is missing from the public audit')\n\n"
        "for channel in ('C1', 'C2', 'C3'):\n"
        "    assert family_from_audit(equation_parameters[channel], 'adstock') == 'geometric'\n"
        "    assert family_from_audit(equation_parameters[channel], 'saturation') == 'michaelis_menten'\n"
        "assert easy_cfg.confounding_strength_range == (0.0, 0.0)\n\n"
        "expected_exogenous_shapes = {\n"
        "    'eps_d': (T_full, world.J), 'eps_z': (T_full, world.M),\n"
        "    'eps_c': (T_full, world.K), 'eps_b': (T_full,), 'eps_y': (T_full,),\n"
        "    'eps_c_hf': (T_full, world.K), 'eps_c_pulse': (T_full, world.K),\n"
        "}\n"
        "exogenous_shapes = {name: values.shape for name, values in exogenous.items()}\n"
        "assert exogenous_shapes == expected_exogenous_shapes\n"
        "print('\\nfull-horizon exogenous shapes:', exogenous_shapes)\n"
        "print('Exact replay needs these innovations together with the realized equation parameters.')"
    ),
    code(
        "def show_fig(plot_fn, title):\n"
        "    path = os.path.join(tempfile.mkdtemp(), 'fig.png')\n"
        "    plot_fn(world, path, title)\n"
        "    display(Image(filename=path))\n\n"
        "show_fig(viz.plot_dag, 'Causal graph: Z controls enter through baseline B only')"
    ),
    md(
        "The next graph is the PyMC **generative** model built from this world's recorded graph "
        "and structural choices. It is not the posterior oracle. The dependency view is curated "
        "to the main generated outputs so the graph remains readable; PyMC includes their "
        "upstream random variables automatically."
    ),
    code(
        "try:\n"
        "    import graphviz\n"
        "except ImportError as exc:\n"
        "    raise RuntimeError('Install the docs extra to render this model graph.') from exc\n"
        "if shutil.which('dot') is None:\n"
        "    raise RuntimeError('Install the system Graphviz package to render this model graph.')\n\n"
        "model, _, _ = pg.build_world_model(\n"
        "    world.g, world.cfg, world.extras['structural'], world.T,\n"
        "    prior_cond=world.extras.get('prior_cond'),\n"
        ")\n"
        "var_names = ['demand', 'controls', 'channels', 'baseline', 'contributions', 'sales']\n"
        "available = [name for name in var_names if name in model.named_vars]\n"
        "print('generative model variables shown:', ', '.join(available))\n"
        "model_graph = pm.model_to_graphviz(model, var_names=available)\n"
        "model_graph.graph_attr.update(size='12,7!', ratio='fill')\n"
        "display(model_graph)"
    ),
    md(
        "The generated time-series plot shows spend, observed controls, latent demand, and sales. "
        "D1 is displayed for auditing even though it is hidden and isolated. The channel plot "
        "compares each channel's spend with its true geometric-adstocked, "
        "Michaelis-Menten-saturated effect. The final plot shows the exact decomposition, whose "
        "reconstruction is verified numerically."
    ),
    code(
        "show_fig(viz.plot_timeseries, 'Generated observables: spend, controls, and sales')\n"
        "show_fig(viz.plot_channels, 'Per-channel spend and true media effects')\n"
        "show_fig(viz.plot_decomposition, 'Exact sales decomposition')\n"
        "print('world reconstruction max abs error:', f'{world.identity_error():.2e}')"
    ),
    md(
        "## Test effect recovery with a regular PyMC-Marketing MMM\n\n"
        "The graph above satisfies the standard adjustment story for a regression on the "
        "observed variables: all three channels enter `Y` directly, both observed controls "
        "enter through `B`, and the latent `D1` node is isolated. We now give the installed "
        "PyMC-Marketing `MMM` only the dated `C1`–`C3`, `Z1`–`Z2`, and `Y` columns. The fit "
        "does not receive the true graph, coefficients, baseline, contributions, or innovations.\n\n"
        "The model uses the matching geometric-adstock and Michaelis-Menten component classes "
        "with their installed default priors. Every other model option stays at its default: "
        "in particular, the intercept is constant and there is no seasonality. The sampler "
        "changes only `chains=4` and `draws=500`; default tuning is retained. The fixed random "
        "seed changes no statistical setting and makes the executable result reproducible."
    ),
    code(
        "CHANNEL_COLUMNS = [f'C{k + 1}' for k in range(world.K)]\n"
        "CONTROL_COLUMNS = [f'Z{m + 1}' for m in range(world.M)]\n"
        "mmm_dataset = pd.DataFrame({'date': pd.date_range('2025-01-06', periods=world.T, freq='W-MON')})\n"
        "for k, name in enumerate(CHANNEL_COLUMNS):\n"
        "    mmm_dataset[name] = world.data['channels'][:, k]\n"
        "for m, name in enumerate(CONTROL_COLUMNS):\n"
        "    mmm_dataset[name] = world.data['controls'][:, m]\n"
        "mmm_dataset['Y'] = world.data['sales']\n\n"
        "weeks = np.arange(world.T)\n"
        "RECOVERY_START = easy_cfg.l_max\n"
        "assert list(mmm_dataset) == ['date', *CHANNEL_COLUMNS, *CONTROL_COLUMNS, 'Y']\n"
        "display(mmm_dataset.head())"
    ),
    code(
        "X_mmm = mmm_dataset.drop(columns='Y')\n"
        "y_mmm = mmm_dataset['Y']\n"
        "mmm = MMM(\n"
        "    date_column='date',\n"
        "    channel_columns=CHANNEL_COLUMNS,\n"
        "    control_columns=CONTROL_COLUMNS,\n"
        "    target_column='Y',\n"
        "    adstock=GeometricAdstock(l_max=easy_cfg.l_max),\n"
        "    saturation=MichaelisMentenSaturation(),\n"
        ")\n"
        "mmm.build_model(X_mmm, y_mmm)\n"
        "mmm.add_original_scale_contribution_variable(\n"
        "    ['channel_contribution', 'control_contribution']\n"
        ")\n"
        "idata = mmm.fit(X_mmm, y_mmm, chains=4, draws=500, random_seed=20260726)\n"
        "posterior = idata['/posterior']\n"
        "divergences = int(idata['/sample_stats']['diverging'].sum())\n"
        "print('posterior draws:', posterior.sizes['chain'] * posterior.sizes['draw'])\n"
        "print('divergences:', divergences)"
    ),
    md(
        "Recovery is evaluated on the original sales scale after the first `l_max` weeks. The "
        "generated truth includes pre-window adstock history that the fitted MMM never sees, "
        "so scoring that warm-up region would compare different estimands. For every channel "
        "and control, the table compares the known post-warm-up cumulative effect with its "
        "posterior interval and compares the posterior-mean effect path with the known weekly "
        "path. `covered_94` asks whether the true cumulative effect falls inside the equal-tailed "
        "94% posterior interval; `curve_correlation` and `relative_curve_rmse` assess temporal "
        "recovery."
    ),
    code(
        "def driver_recovery(samples, truth, dimension, names, kind):\n"
        "    samples = samples.isel(date=slice(RECOVERY_START, None))\n"
        "    truth = truth[RECOVERY_START:]\n"
        "    total_quantiles = samples.sum('date').quantile([0.03, 0.5, 0.97], dim=('chain', 'draw'))\n"
        "    posterior_mean = samples.mean(('chain', 'draw'))\n"
        "    rows = []\n"
        "    for index, name in enumerate(names):\n"
        "        true_curve = np.asarray(truth[:, index])\n"
        "        estimated_curve = np.asarray(posterior_mean.sel({dimension: name}))\n"
        "        true_total = float(true_curve.sum())\n"
        "        low = float(total_quantiles.sel({dimension: name, 'quantile': 0.03}))\n"
        "        median = float(total_quantiles.sel({dimension: name, 'quantile': 0.5}))\n"
        "        high = float(total_quantiles.sel({dimension: name, 'quantile': 0.97}))\n"
        "        rows.append({\n"
        "            'driver': name, 'kind': kind, 'true_total': true_total,\n"
        "            'posterior_median_total': median, 'low_94': low, 'high_94': high,\n"
        "            'covered_94': low <= true_total <= high,\n"
        "            'curve_correlation': np.corrcoef(true_curve, estimated_curve)[0, 1],\n"
        "            'relative_curve_rmse': (\n"
        "                np.sqrt(np.mean((estimated_curve - true_curve) ** 2))\n"
        "                / (np.std(true_curve) + 1e-12)\n"
        "            ),\n"
        "        })\n"
        "    return pd.DataFrame(rows)\n\n"
        "channel_samples = posterior['channel_contribution_original_scale']\n"
        "control_samples = posterior['control_contribution_original_scale']\n"
        "recovery = pd.concat([\n"
        "    driver_recovery(\n"
        "        channel_samples, world.data['contributions_observed'],\n"
        "        'channel', CHANNEL_COLUMNS, 'channel',\n"
        "    ),\n"
        "    driver_recovery(\n"
        "        control_samples, world.data['control_contribution'],\n"
        "        'control', CONTROL_COLUMNS, 'control',\n"
        "    ),\n"
        "], ignore_index=True)\n"
        "display(recovery.round(3))\n"
        "channel_recovery = recovery.query(\"kind == 'channel'\")\n"
        "recovery_pass = bool(\n"
        "    divergences == 0\n"
        "    and channel_recovery['covered_94'].all()\n"
        "    and (channel_recovery['curve_correlation'] >= 0.8).all()\n"
        ")\n"
        "print('channel cumulative effects covered:', f\"{channel_recovery['covered_94'].mean():.0%}\")\n"
        "print('requested default-fit recovery verdict:', 'PASS' if recovery_pass else 'FAIL')"
    ),
    code(
        "channel_quantiles = channel_samples.quantile(\n"
        "    [0.03, 0.5, 0.97], dim=('chain', 'draw')\n"
        ")\n"
        "scored_weeks = weeks[RECOVERY_START:]\n"
        "fig, axes = plt.subplots(world.K, 1, figsize=(10, 7), sharex=True)\n"
        "for k, (ax, name) in enumerate(zip(axes, CHANNEL_COLUMNS, strict=True)):\n"
        "    true_effect = world.data['contributions_observed'][RECOVERY_START:, k]\n"
        "    channel_curve = channel_quantiles.sel(channel=name).isel(\n"
        "        date=slice(RECOVERY_START, None)\n"
        "    )\n"
        "    low = channel_curve.sel(quantile=0.03)\n"
        "    median = channel_curve.sel(quantile=0.5)\n"
        "    high = channel_curve.sel(quantile=0.97)\n"
        "    ax.plot(scored_weeks, true_effect, color='black', lw=1.5, label='known effect')\n"
        "    ax.plot(scored_weeks, median, color='C0', lw=1.3, label='posterior median')\n"
        "    ax.fill_between(\n"
        "        scored_weeks, low, high, color='C0', alpha=0.2, label='94% interval'\n"
        "    )\n"
        "    ax.set(ylabel=name, title=f'{name}: known versus recovered weekly contribution')\n"
        "    ax.spines[['top', 'right']].set_visible(False)\n"
        "axes[0].legend(frameon=False, ncol=3)\n"
        "axes[-1].set_xlabel('week')\n"
        "fig.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "### Interpretation of the executed recovery check\n\n"
        "This check intentionally reports the result rather than assuming the desired answer. "
        "For the committed seed, the requested default fit returns `FAIL`: divergent NUTS "
        "transitions make the posterior unreliable, and the known cumulative channel effects "
        "are not all recovered. The controls and the shapes of some contribution paths can be "
        "estimated well, but that is weaker than recovering absolute media attribution.\n\n"
        "The graph's back-door adjustment is necessary for causal identification, but it does "
        "not guarantee finite-sample parameter recovery. This world also has an unobserved "
        "random-walk baseline, 48 observations, channels that remain positive rather than "
        "providing repeated off periods, and adstock/saturation parameters that trade off with "
        "the intercept. A genuinely easy recovery benchmark therefore needs additional overlap "
        "or interventions, a baseline model aligned with the generator, longer histories, or "
        "regularizing priors. None is silently added here because the requested fit keeps the "
        "installed model and sampler defaults."
    ),
    md(
        "## Generate both corpora\n\n"
        "`sample_prior_predictive` is the corpus API and creates padded variable-size tasks. "
        "The single full-size world above used `sample_scm` only for its convenient visual API."
    ),
    code(
        "easy = pg.sample_prior_predictive(easy_cfg)\n"
        "hard = pg.sample_prior_predictive(hard_cfg)\n"
        "assert np.array_equal(easy['g'], hard['g'])\n"
        "assert np.array_equal(easy['adstock_family'], hard['adstock_family'])\n"
        "for name, corpus in [('easy', easy), ('difficult', hard)]:\n"
        "    errors = DataGenerator.validate_corpus(corpus)\n"
        "    assert errors == [], errors\n"
        '    print(f\'{name}: N={len(corpus["sales_raw"])} elapsed={corpus["diagnostics"]["elapsed_s"]:.2f}s\')'
    ),
    md(
        "## Read the active masks\n\nPrefix masks identify active slots. The table shows observed cells with channels and controls switched off by padding."
    ),
    code(
        "def count_table(corpus):\n"
        "    return [\n"
        "        (int(corpus['cell_id'][i]), int(corpus['K_active'][i]), int(corpus['M_active'][i]),\n"
        "         ''.join(map(str, corpus['active_c_mask'][i])),\n"
        "         ''.join(map(str, corpus['active_m_mask'][i])))\n"
        "        for i in range(len(corpus['cell_id']))\n"
        "    ]\n\n"
        "print('cell  K  M  active_C  active_Z')\n"
        "for row in count_table(easy):\n"
        "    print(f'{row[0]:>4} {row[1]:>2} {row[2]:>2} {row[3]:>9} {row[4]:>9}')\n"
        "assert np.any(easy['K_active'] < 3) and np.any(easy['M_active'] < 2)\n"
        "assert np.all(easy['spend_raw'] * (1 - easy['active_c_mask'][:, None, :]) == 0)\n"
        "assert np.all(easy['controls'] * (1 - easy['active_m_mask'][:, None, :]) == 0)"
    ),
    md(
        "## Audit every graph and padded mechanism field\n\nThe latent demand node is active only as a padded-layout convention: it has no incident edges and its contributions are exactly zero."
    ),
    code(
        "def assert_simple_corpus(corpus, cfg):\n"
        "    layout = SlotLayout(3, 2, 1, edge_types=cfg.layout.edge_types)\n"
        "    blocks = layout.unpack(corpus['g'])\n"
        "    c_mask, m_mask, j_mask = (corpus['active_c_mask'], corpus['active_m_mask'], corpus['active_j_mask'])\n"
        "    assert np.array_equal(c_mask, np.arange(3)[None, :] < corpus['K_active'][:, None])\n"
        "    assert np.array_equal(m_mask, np.arange(2)[None, :] < corpus['M_active'][:, None])\n"
        "    assert np.array_equal(j_mask, np.ones((len(c_mask), 1), dtype=np.uint8))\n"
        "    assert np.all(corpus['J_active'] == 1)\n"
        "    assert not blocks['dc'].any() and not blocks['dz'].any() and not blocks['db'].any()\n"
        "    assert not blocks['zc'].any() and not blocks['cc'].any() and not blocks['zz'].any()\n"
        "    assert np.array_equal(blocks['cy'], c_mask)\n"
        "    assert np.array_equal(blocks['zb'], m_mask)\n"
        "    active_adstock = np.take(ADSTOCK_FAMILY_KEYS, corpus['adstock_family'][c_mask.astype(bool)])\n"
        "    inactive_adstock = np.take(ADSTOCK_FAMILY_KEYS, corpus['adstock_family'][~c_mask.astype(bool)])\n"
        "    assert np.all(active_adstock == 'geometric')\n"
        "    assert np.all(inactive_adstock == 'none')\n"
        "    for key in ('spend_raw', 'contributions_raw', 'channel_level', 'saturation_scale',\n"
        "                'adstock_alpha', 'weibull_lam', 'weibull_k'):\n"
        "        values = corpus[key]\n"
        "        assert np.all(values * (1 - c_mask[:, None, :] if values.ndim == 3 else 1 - c_mask) == 0), key\n"
        "    assert np.all(corpus['controls'] * (1 - m_mask[:, None, :]) == 0)\n"
        "    assert np.all(corpus['demand'] * (1 - j_mask[:, None, :]) == 0)\n"
        "    assert np.all(corpus['confounder_contribution'] == 0)\n"
        "    assert np.all(corpus['confounding_strength'] == 0)\n"
        "    assert np.all(corpus['indirect_effects'] == 0)\n"
        "    assert np.all(corpus['indirect_effects_by_source'] == 0)\n"
        "    additive = corpus['baseline_raw'] + corpus['contributions_raw'].sum(-1) + corpus['indirect_effects']\n"
        "    full = (corpus['baseline_intrinsic'] + corpus['confounder_contribution'].sum(-1) +\n"
        "            corpus['control_contribution'].sum(-1) +\n"
        "            corpus['contributions_raw'].sum(-1) + corpus['indirect_effects_by_source'].sum(-1))\n"
        "    assert np.allclose(corpus['sales_raw'], additive, atol=2e-6)\n"
        "    assert np.allclose(corpus['sales_raw'], full, atol=2e-6)\n"
        "    assert np.allclose(corpus['indirect_effects'], corpus['indirect_effects_by_source'].sum(-1), atol=2e-6)\n\n"
        "assert_simple_corpus(easy, easy_cfg)\n"
        "assert_simple_corpus(hard, hard_cfg)\n"
        "print('all graph, padding, isolation, and decomposition assertions passed')"
    ),
    md(
        "## Family audit\n\nThe corpus persists active-channel `adstock_family` values, but its unchanged "
        "schema does not persist `sat_family`. The named configuration and the public audit of the "
        "single sampled world establish the selected Michaelis-Menten saturation without relying on "
        "a memorized categorical order."
    ),
    code(
        "active_adstock = np.take(ADSTOCK_FAMILY_KEYS, easy['adstock_family'][easy['active_c_mask'].astype(bool)])\n"
        "assert set(active_adstock) == {'geometric'}\n"
        "assert easy_cfg.adstock_family_probs['geometric'] == 1.0\n"
        "assert easy_cfg.saturation_family_probs['michaelis_menten'] == 1.0\n"
        "print('corpus active-channel adstock:', sorted(set(active_adstock)))\n"
        "print('single-world saturation audit: Michaelis-Menten (shown above)')"
    ),
    md(
        "## Easy versus difficult: a noise diagnostic\n\nThe statistic below is the standard deviation of the persisted intrinsic baseline after the control and latent baseline terms have been removed. In this configuration the latent term is exactly zero. The intrinsic baseline contains the baseline walk and the observed-sales noise walk, so changing only `rw_sales_std_sigma` should increase this diagnostic. It is a generator diagnostic, not a causal estimator. The existing signal labels describe channel-target texture, so they are reported only as supplementary context."
    ),
    code(
        "def noise_summary(corpus):\n"
        "    signal = corpus['identifiability']['signal_metrics']\n"
        "    valid = corpus['identifiability']['signal_metric_valid'].astype(bool)\n"
        "    spearman_index = pg.SIGNAL_METRIC_LAYOUT.index('spearman')\n"
        "    intrinsic_std = corpus['baseline_intrinsic'].std(axis=1).mean()\n"
        "    return float(intrinsic_std), float(signal[..., spearman_index][valid[..., spearman_index]].mean())\n\n"
        "easy_noise, easy_spearman = noise_summary(easy)\n"
        "hard_noise, hard_spearman = noise_summary(hard)\n"
        "print('variant    rw_sales_std_sigma  intrinsic_std  mean signal spearman')\n"
        "print(f'easy       {easy_cfg.rw_sales_std_sigma:>18.2f}  {easy_noise:>12.4f}  {easy_spearman:>20.4f}')\n"
        "print(f'difficult  {hard_cfg.rw_sales_std_sigma:>18.2f}  {hard_noise:>12.4f}  {hard_spearman:>20.4f}')\n"
        "assert hard_noise > easy_noise\n"
        "print(f'intrinsic-baseline ratio: {hard_noise / easy_noise:.1f}x')"
    ),
    md(
        "The rows below are paired by the shared seed, cell, and draw order. The assertions verify "
        "that their graph, spend, and controls are identical, so the displayed sales and baseline "
        "difference is attributable to the only changed generator knob: `rw_sales_std_sigma`."
    ),
    code(
        "row = 0\n"
        "assert np.array_equal(easy['g'][row], hard['g'][row])\n"
        "assert np.array_equal(easy['spend_raw'][row], hard['spend_raw'][row])\n"
        "assert np.array_equal(easy['controls'][row], hard['controls'][row])\n"
        "assert np.array_equal(easy['contributions_raw'][row], hard['contributions_raw'][row])\n"
        "weeks = np.arange(easy_cfg.T)\n"
        "fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)\n"
        "for corpus, label in ((easy, 'easy'), (hard, 'difficult')):\n"
        "    axes[0].plot(weeks, corpus['sales_raw'][row], lw=1.4, label=label)\n"
        "    axes[1].plot(weeks, corpus['baseline_raw'][row], lw=1.4, label=label)\n"
        "axes[0].set(title='Matched corpus row: observed sales', ylabel='sales')\n"
        "axes[1].set(title='Matched corpus row: baseline includes observed-sales noise', xlabel='week', ylabel='baseline')\n"
        "for ax in axes:\n"
        "    ax.legend(frameon=False)\n"
        "    ax.spines[['top', 'right']].set_visible(False)\n"
        "fig.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "## Recap\n\nThis is a small, reproducible padded corpus with a controlled graph and mechanism family. Increasing only `rw_sales_std_sigma` makes the observed sales path harder without introducing latent or shared baseline-channel confounding."
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
