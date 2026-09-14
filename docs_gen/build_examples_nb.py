"""Assemble ``docs/examples/index.ipynb`` from ordered cells with nbformat.

Run from the repo root:  python docs_gen/build_examples_nb.py

The notebook is a data-generation tour: it starts from the smallest graph the
generator can draw, introduces the parameter groups one at a time, and ends on a
confounded multi-channel world with interacting controls. It only imports
``prior_generator``, so a reader can download it and run it as-is. It is executed
at docs-build time by mkdocs-jupyter.
"""

from __future__ import annotations

import os

import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md(
        "# Generating data — from the smallest graph to a confounded world\n"
        "\n"
        "`prior_generator` draws **synthetic marketing worlds** as structural causal\n"
        "models: a DAG over media channels `C`, observed controls `Z`, hidden demand\n"
        "factors `D`, a baseline `B`, and sales `Y`, with an exact decomposition of\n"
        "`Y` into its causes.\n"
        "\n"
        "This notebook builds up one knob at a time:\n"
        "\n"
        "| section | what it adds |\n"
        "| --- | --- |\n"
        "| 1 | the minimal world: one channel, nothing else |\n"
        "| 2 | what a world contains — observables versus truth |\n"
        "| 3 | the parameter groups, and where each one enters |\n"
        "| 4 | media response: adstock and saturation families |\n"
        "| 5 | channel texture — what makes spend sweep its response curve |\n"
        "| 6 | outcome-side noise and a descriptive difficulty ratio |\n"
        "| 7 | controls and interactions: `Z→Y`, `Z→C`, `C→C` |\n"
        "| 8 | hidden confounding: `D→C` with `D→Y` |\n"
        "| 9 | a corpus of many worlds, validated |\n"
        "| 10 | how large is everything? outcome distributions across worlds |\n"
        "\n"
        "> Every cell below is executed when the docs are built — the outputs and\n"
        "> figures you see are real."
    ),
    code(
        "%matplotlib inline\n"
        "\n"
        "import os\n"
        "import tempfile\n"
        "\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from IPython.display import Image, display\n"
        "\n"
        "import prior_generator as pg\n"
        "from prior_generator import viz\n"
        "\n"
        "print('prior-generator', pg.__version__)"
    ),
    # ---------------------------------------------------------------- 1
    md(
        "## 1 · The minimal world\n"
        "\n"
        "Every world is drawn from an `SCMPrior`. Build one with `make_scm_prior`:\n"
        "the three sizes are **required** because they pin the tensor layout, and\n"
        "`edge_budget` says how many arrows of each type to place.\n"
        "\n"
        "The smallest useful graph is one channel affecting sales and nothing else.\n"
        "`edge_budget={'cy': (1, 1)}` means *exactly one* `C→Y` arrow; every other\n"
        "edge type is pinned to zero, so there are no controls, no confounding, and\n"
        "no channel interactions.\n"
        "\n"
        "The eight edge types are `cy` (channel→sales), `zy` (control→sales),\n"
        "`dy` (demand→sales), `dc` (demand→channel), `dz` (demand→control),\n"
        "`zc` (control→channel), `cc` (channel→channel), and `zz`\n"
        "(control→control)."
    ),
    code(
        "MINIMAL_EDGES = {\n"
        "    'cy': (1, 1),   # exactly one channel -> sales arrow\n"
        "    'zy': (0, 0), 'dy': (0, 0), 'dc': (0, 0), 'dz': (0, 0),\n"
        "    'zc': (0, 0), 'cc': (0, 0), 'zz': (0, 0),\n"
        "}\n"
        "\n"
        "minimal = pg.make_scm_prior(\n"
        "    n_treatments=1,       # media channels (the interventions)\n"
        "    n_covariates=1,       # observed controls\n"
        "    n_latent=1,           # hidden demand factors\n"
        "    n_time_steps=104,     # weeks\n"
        "    edge_budget=MINIMAL_EDGES,\n"
        "    n_cells=2, draws_per_cell=1, seed=20260728,\n"
        ")\n"
        "\n"
        "world = pg.sample_scm(minimal, seed=1, name='minimal')\n"
        "print(f'n_time_steps={world.n_time_steps}  n_treatments={world.n_treatments}  '\n"
        "      f'n_covariates={world.n_covariates}  n_latent={world.n_latent}')\n"
        "print('edges present:', {k: int(np.asarray(v).sum()) for k, v in world.g.items()})"
    ),
    md(
        "`sample_scm` returns one accepted world. `n_time_steps` is the reported\n"
        "horizon; the graph is simulated over `n_time_steps + adstock_burn_in` weeks\n"
        "and sliced, so the first reported week already has real carryover history\n"
        "behind it."
    ),
    code(
        "def show(plot_fn, world, title):\n"
        '    """Render one of the viz helpers inline."""\n'
        "    path = os.path.join(tempfile.mkdtemp(), 'fig.png')\n"
        "    plot_fn(world, path, title)\n"
        "    display(Image(filename=path))\n"
        "\n"
        "show(viz.plot_dag, world, 'The minimal graph: C1 -> Y, nothing else')"
    ),
    # ---------------------------------------------------------------- 2
    md(
        "## 2 · What a world contains\n"
        "\n"
        "`world.data` holds both what a model may see and what only the generator\n"
        "knows. Keeping the two apart is the whole point of the package.\n"
        "\n"
        "**Model-facing** (`channels`, `controls`, `sales`) is what an MMM gets.\n"
        "Everything else is **truth**, for scoring only: `contributions_observed`\n"
        "(each channel's true weekly effect on the realised spend path),\n"
        "`contributions` (the same response on the *counterfactual* spend with all\n"
        "upstream arrows cut), `baseline`, `demand`, and the per-source splits."
    ),
    code(
        "MODEL_FACING = ('channels', 'controls', 'sales')\n"
        "for name, array in sorted(world.data.items()):\n"
        "    array = np.asarray(array)\n"
        "    if array.ndim == 0 or array.size == 0:\n"
        "        continue\n"
        "    tag = 'observable' if name in MODEL_FACING else 'truth'\n"
        "    print(f'{name:28s} {str(array.shape):12s} {tag}')"
    ),
    md(
        "The decomposition is **exact**, not approximate. Sales equals the sum of its\n"
        "causes to floating-point error, and `identity_error()` proves it on every\n"
        "world:\n"
        "\n"
        "```\n"
        "sales = baseline_intrinsic             (the intercept B)\n"
        "      + sales_noise                    (iid observation noise)\n"
        "      + Σ confounder_contribution      (D → Y)\n"
        "      + Σ control_contribution         (Z → Y)\n"
        "      + Σ contributions                (direct media effect)\n"
        "      + Σ indirect_effects_by_source   (media moved by upstream causes)\n"
        "```"
    ),
    code(
        "print('max |Σ causes − sales| =', f'{world.identity_error():.2e}')\n"
        "assert world.identity_error() < 1e-9"
    ),
    # ---------------------------------------------------------------- 3
    md(
        "## 3 · The parameters\n"
        "\n"
        "`world.params` is the realised draw. It groups into five families, and the\n"
        "`SCMPrior` field that governs each one is named alongside:\n"
        "\n"
        "| group | parameters | prior field |\n"
        "| --- | --- | --- |\n"
        "| **edge coefficients** | `beta` (`C→Y`), `rho_zy` (`Z→Y`), `delta_dy` (`D→Y`), `w_dc` (`D→C`), `u_dz` (`D→Z`), `v_zc` (`Z→C`), `alpha_cc` (`C→C`), `gamma_zz` (`Z→Z`) | `*_coeff_range`, `beta_additive_range` |\n"
        "| **media response** | `adstock_family`, `adstock_alpha`, `weibull_*`, `sat_family`, `mm_kappa_mult`, `hill_*`, `logistic_lam`, `tanh_c`, `root_alpha` | `*_family_probs`, `adstock_alpha_range` |\n"
        "| **node processes** | `rw_d`, `rw_z`, `rw_c`, `rw_b` random walks, `rw_y` iid noise | `rw_*_range`, `rw_*_sigma`, `rw_smoothness_*` |\n"
        "| **channel texture** | `hf_sigma`, `pulse_amp`, `pulse_prob` | `channel_hf_sigma_range`, `channel_pulse_*_range` |\n"
        "| **control texture** | `control_hf_sigma`, `control_pulse_amp`, `control_pulse_prob` (pulse centred) | `control_hf_sigma_range`, `control_pulse_*_range` |\n"
        "| **anchors** | `channel_level`, `saturation_scale` | derived from the above |\n"
        "\n"
        "`world.equations` renders the executed structural assignments for this exact\n"
        "world, and `world.equation_parameters` gives the realised numbers behind\n"
        "them — so a world is auditable without reading the source."
    ),
    code("for node in ('C1', 'B', 'Y'):\n    print(f'{node}:  {world.equations[node]}\\n')"),
    code(
        "def fmt(group, places=3):\n"
        '    """Readable one-line view of a walk / noise parameter group."""\n'
        "    out = {}\n"
        "    for key, value in group.items():\n"
        "        array = np.asarray(value)\n"
        "        out[key] = (np.round(array, places) if array.dtype.kind in 'fc' else array).tolist()\n"
        "    return out\n"
        "\n"
        "params = world.params\n"
        "print('beta  (C->Y effect):', np.round(np.asarray(params['beta']), 3))\n"
        "print('rw_c  (spend walk) :', fmt(params['rw_c']))\n"
        "print('rw_b  (baseline walk):', fmt(params['rw_b']))\n"
        "print('rw_y  (iid sales noise):', fmt(params['rw_y'], 4))"
    ),
    code(
        "# Only the shape params this world's own families consume carry meaning.\n"
        "from prior_generator.sampler import ADSTOCK_FAMILY_KEYS, SATURATION_FAMILY_KEYS\n"
        "from prior_generator.world_model import (\n"
        "    ADSTOCK_FAMILY_PARAM_NAMES,\n"
        "    SATURATION_FAMILY_PARAM_NAMES,\n"
        ")\n"
        "\n"
        "adstock = ADSTOCK_FAMILY_KEYS[int(params['adstock_family'][0])]\n"
        "saturation = SATURATION_FAMILY_KEYS[int(params['sat_family'][0])]\n"
        "print(f'C1 drew adstock={adstock!r}, saturation={saturation!r}')\n"
        "for name in ADSTOCK_FAMILY_PARAM_NAMES[adstock] + SATURATION_FAMILY_PARAM_NAMES[saturation]:\n"
        "    print(f'  {name:16s}', np.round(np.asarray(params[name]), 3))\n"
        "print(f\"  {'saturation_scale':16s}\", np.round(np.asarray(world.data['saturation_scale']), 3))"
    ),
    md(
        "Note `rw_y` has a `std` but **no smoothness**: the outcome's exogenous term\n"
        "is iid observation noise. Every other node carries a smoothed random walk,\n"
        "whose `smoothness` sets its moving-average width in weeks. `B` is therefore\n"
        "the only latent *trend* in the outcome, which is what makes the baseline and\n"
        "the noise separately identified."
    ),
    # ---------------------------------------------------------------- 4
    md(
        "## 4 · Media response: adstock × saturation\n"
        "\n"
        "Each channel's effect is `beta * saturation(adstock(spend))`. Both transforms\n"
        "come from `pymc-marketing`, so a standard MMM can represent them exactly.\n"
        "Families are drawn per channel from `adstock_family_probs` and\n"
        "`saturation_family_probs`; pin them to a single name for a controlled study.\n"
        "\n"
        "Saturation is **κ-relative**: the knee sits at\n"
        "`kappa_mult × saturation_scale`, where `saturation_scale` is the channel's\n"
        "expected spend level computed from parameters alone. So a channel operates\n"
        "near its own knee whatever its spend units are."
    ),
    code(
        "print('adstock families   :', ADSTOCK_FAMILY_KEYS)\n"
        "print('saturation families:', SATURATION_FAMILY_KEYS)\n"
        "\n"
        "def pin(adstock, saturation):\n"
        '    """Family probability dicts that select exactly one family each."""\n'
        "    return {\n"
        "        'adstock_family_probs': {**dict.fromkeys(ADSTOCK_FAMILY_KEYS, 0.0), adstock: 1.0},\n"
        "        'saturation_family_probs': {\n"
        "            **dict.fromkeys(SATURATION_FAMILY_KEYS, 0.0), saturation: 1.0\n"
        "        },\n"
        "    }\n"
        "\n"
        "print('\\nper-family shape params:')\n"
        "for family, names in SATURATION_FAMILY_PARAM_NAMES.items():\n"
        "    print(f'  {family:18s} {names or \"(none)\"}')"
    ),
    code(
        "# The response curve each family implies, on the same adstocked spend.\n"
        "import pytensor.tensor as pt\n"
        "from prior_generator import mechanisms\n"
        "\n"
        "grid = np.linspace(0.0, 4.0, 200)\n"
        "shape_args = {\n"
        "    'hill': dict(slope=2.0, kappa_mult=1.0),\n"
        "    'logistic': dict(lam=1.5),\n"
        "    'michaelis_menten': dict(kappa_mult=1.0),\n"
        "    'tanh': dict(c=0.8),\n"
        "    'root': dict(alpha=0.6),\n"
        "}\n"
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "for name, kwargs in shape_args.items():\n"
        "    curve = mechanisms.SATURATION_FAMILIES[name](\n"
        "        pt.as_tensor_variable(grid), pt.as_tensor_variable(1.0), **kwargs\n"
        "    ).eval()\n"
        "    ax.plot(grid, curve, lw=1.6, label=name)\n"
        "ax.axvline(1.0, color='0.6', ls=':', lw=1)\n"
        "ax.set(xlabel='adstocked spend / saturation_scale', ylabel='response',\n"
        "       title='Saturation families at scale = 1 (dotted line = the anchor)')\n"
        "ax.legend(frameon=False)\n"
        "ax.spines[['top', 'right']].set_visible(False)\n"
        "plt.show()"
    ),
    # ---------------------------------------------------------------- 5
    md(
        "## 5 · Channel texture\n"
        "\n"
        "A saturating, carryover-smoothed response can only be *identified* if spend\n"
        "actually moves across the curve. Three knobs drive that, all relative to the\n"
        "channel's own level so they are scale-free:\n"
        "\n"
        "- `rw_channel_std_range` — the slow walk in spend,\n"
        "- `channel_hf_sigma_range` — iid weekly execution noise,\n"
        "- `channel_pulse_prob_range` / `channel_pulse_amp_range` — campaign bursts.\n"
        "\n"
        "Compare a deliberately smooth channel against the shipped default."
    ),
    code(
        "def one_channel(label, **overrides):\n"
        "    cfg = pg.make_scm_prior(\n"
        "        n_treatments=1, n_covariates=1, n_latent=1, n_time_steps=104,\n"
        "        edge_budget=MINIMAL_EDGES, n_cells=2, draws_per_cell=1, seed=20260728,\n"
        "        **pin('geometric', 'michaelis_menten'), **overrides,\n"
        "    )\n"
        "    return pg.sample_scm(cfg, seed=4, name=label)\n"
        "\n"
        "smooth = one_channel('smooth', channel_hf_sigma_range=(0.0, 0.0),\n"
        "                     channel_pulse_prob_range=(0.0, 0.0),\n"
        "                     rw_channel_std_range=(0.15, 0.2))\n"
        "textured = one_channel('textured')   # shipped default texture\n"
        "\n"
        "fig, axes = plt.subplots(2, 2, figsize=(11, 5), sharex=True)\n"
        "for col, (w, label) in enumerate(((smooth, 'smooth'), (textured, 'default texture'))):\n"
        "    weeks = np.arange(w.n_time_steps)\n"
        "    spend = w.data['channels'][:, 0]\n"
        "    axes[0, col].plot(weeks, spend, lw=1.2, color='C0')\n"
        "    axes[0, col].set_title(f'{label}: spend  (CV {spend.std() / spend.mean():.2f})')\n"
        "    axes[1, col].plot(weeks, w.data['contributions_observed'][:, 0], lw=1.2, color='C1')\n"
        "    axes[1, col].set(xlabel='week')\n"
        "    axes[1, col].set_title('true contribution')\n"
        "for ax in axes.ravel():\n"
        "    ax.spines[['top', 'right']].set_visible(False)\n"
        "fig.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "Texture creates variation that can help estimate response shape and carryover.\n"
        "It does not by itself establish identifiability: input dependence, observation\n"
        "noise, model assumptions, and the observed horizon also matter."
    ),
    # ---------------------------------------------------------------- 6
    md(
        "## 6 · Outcome-side noise and a difficulty diagnostic\n"
        "\n"
        "Two terms sit between the media effect and observed sales: the baseline walk\n"
        "`RW_B` (a latent trend) and `RW_Y` (iid observation noise). Their amplitudes\n"
        "are drawn *relative to the media amplitude*\n"
        "`sqrt(Σ (g_cy · beta)²)` — a function of parameters alone — so the\n"
        "signal-to-noise ratio is a declared axis rather than an accident of units:\n"
        "\n"
        "- `rw_baseline_std_range` — the latent trend, in media-amplitude units,\n"
        "- `rw_sales_std_range` — the observation noise, same units,\n"
        "- `outcome_std_mode='absolute'` — opt out and set the old absolute scales.\n"
        "\n"
        "One descriptive difficulty measure is\n"
        "\n"
        "$$\\text{ratio} = \\frac{\\mathrm{sd}(\\text{baseline trend} + \\text{latent baseline})}{\\mathrm{sd}(\\text{total media contribution})}$$\n"
        "\n"
        "The settings below are illustrative prior-design choices, not an empirical\n"
        "calibration to real MMM datasets. This ratio omits iid observation noise and\n"
        "cannot certify recoverability. We measure it on a restricted one-channel prior."
    ),
    code(
        "def outcome_ratio(w):\n"
        '    """sd(unrepresentable outcome terms) / sd(total media contribution)."""\n'
        "    start = w.cfg.l_max\n"
        "    residual = (np.asarray(w.data['baseline_intrinsic'])\n"
        "                + np.asarray(w.data['confounder_contribution']).sum(1))[start:]\n"
        "    media = np.asarray(w.data['contributions_observed']).sum(1)[start:]\n"
        "    return float(np.std(residual) / np.std(media))\n"
        "\n"
        "# Measure this restricted prior in one corpus.\n"
        "ratio_cfg = pg.make_scm_prior(\n"
        "    n_treatments=1, n_covariates=1, n_latent=1, n_time_steps=104,\n"
        "    edge_budget=MINIMAL_EDGES, n_cells=12, draws_per_cell=3, seed=20260728,\n"
        "    **pin('geometric', 'michaelis_menten'),\n"
        ")\n"
        "ratio_corpus = pg.sample_prior_predictive(ratio_cfg)\n"
        "window = slice(ratio_cfg.l_max, None)\n"
        "residual = (ratio_corpus['baseline_intrinsic']\n"
        "            + ratio_corpus['confounder_contribution'].sum(-1))[:, window]\n"
        "media = ratio_corpus['contributions_raw'].sum(-1)[:, window]\n"
        "ratios = residual.std(axis=1) / media.std(axis=1)\n"
        "q5, q50, q95 = np.quantile(ratios, [0.05, 0.5, 0.95])\n"
        "print(f'residual/media ratio over {len(ratios)} worlds: '\n"
        "      f'p5 {q5:.2f}   median {q50:.2f}   p95 {q95:.2f}')\n"
        "print('Illustrative simulation diagnostic; not an empirical MMM calibration.')"
    ),
    code(
        "# Turn the knob explicitly: a quiet world and a loud one, same graph.\n"
        "quiet = one_channel('quiet', rw_baseline_std_range=(0.005, 0.010),\n"
        "                    rw_sales_std_range=(0.002, 0.004))\n"
        "loud = one_channel('loud', rw_baseline_std_range=(0.20, 0.30),\n"
        "                   rw_sales_std_range=(0.05, 0.08))\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=False)\n"
        "for ax, w in zip(axes, (quiet, loud)):\n"
        "    weeks = np.arange(w.n_time_steps)\n"
        "    ax.plot(weeks, w.data['sales'], lw=1.2, color='0.25', label='sales')\n"
        "    ax.plot(weeks, w.data['baseline'], lw=1.2, color='C3', label='baseline')\n"
        "    ax.set(xlabel='week', title=f'{w.name}: residual/media = {outcome_ratio(w):.2f}')\n"
        "    ax.legend(frameon=False, fontsize=8)\n"
        "    ax.spines[['top', 'right']].set_visible(False)\n"
        "fig.tight_layout()\n"
        "plt.show()"
    ),
    # ---------------------------------------------------------------- 7
    md(
        "## 7 · Controls and interactions\n"
        "\n"
        "Now grow the graph. Three edge types make the world harder in *structurally*\n"
        "different ways:\n"
        "\n"
        "- **`zy`** — a control affects sales directly. A regression that\n"
        "  includes the control handles this.\n"
        "- **`zc`** — a control also moves spend. Now the control is a confounder of\n"
        "  the media effect, but an **observed** one, so adjusting for it is licensed.\n"
        "- **`cc`** — one channel drives another. The direct effect and the total\n"
        "  effect now differ, which is why the package reports both\n"
        "  `contributions` (direct) and `contributions_observed` (realised)."
    ),
    code(
        "interacting = pg.make_scm_prior(\n"
        "    n_treatments=3, n_covariates=2, n_latent=1, n_time_steps=104,\n"
        "    n_treatments_active_range=(3, 3), n_covariates_active_range=(2, 2),\n"
        "    n_latent_active_range=(1, 1),\n"
        "    edge_budget={'cy': (3, 3), 'zy': (2, 2), 'zc': (2, 2), 'cc': (1, 1),\n"
        "                 'dy': (0, 0), 'dc': (0, 0), 'dz': (0, 0), 'zz': (0, 0)},\n"
        "    n_cells=2, draws_per_cell=1, seed=20260728,\n"
        "    **pin('geometric', 'michaelis_menten'),\n"
        ")\n"
        "mid = pg.sample_scm(interacting, seed=2, name='interacting')\n"
        "show(viz.plot_dag, mid, 'Controls affect sales AND spend; C->C interaction')"
    ),
    code(
        "indirect = np.asarray(mid.data['indirect_effects_by_source'])\n"
        "start = mid.cfg.l_max\n"
        "labels = ('C->C', 'Z->C', 'D->C')\n"
        "print('cumulative effect routed through upstream causes, by source:')\n"
        "for label, column in zip(labels, indirect[start:].T):\n"
        "    print(f'  {label:6s} {column.sum():+10.3f}')\n"
        "print('\\ntotal indirect  ', f\"{np.asarray(mid.data['indirect_effects'])[start:].sum():+10.3f}\")\n"
        "print('total direct    ',\n"
        "      f\"{np.asarray(mid.data['contributions'])[start:].sum():+10.3f}\")"
    ),
    md(
        "The 3-way split is reported under a **fixed sequential zeroing order**\n"
        "(`cc → zc → dc`). The total is order-free and is a genuine estimand; the\n"
        "split is a convention, so do not score a model against it as though it were\n"
        "identified."
    ),
    code("show(viz.plot_decomposition, mid, 'Exact sales decomposition, 3 channels')"),
    # ---------------------------------------------------------------- 8
    md(
        "## 8 · Hidden confounding\n"
        "\n"
        "The hard case. A latent demand factor `D` drives **both** spend (`dc`) and\n"
        "sales (`dy`). Now there is an unobserved common cause of treatment and\n"
        "outcome: a back-door path `C ← D → Y` that no regression on observables\n"
        "can close. Causal sufficiency fails.\n"
        "\n"
        "`D` is in `world.data['demand']` for scoring, and it is **never** a\n"
        "model-facing column."
    ),
    code(
        "confounded_cfg = pg.make_scm_prior(\n"
        "    n_treatments=3, n_covariates=2, n_latent=1, n_time_steps=104,\n"
        "    n_treatments_active_range=(3, 3), n_covariates_active_range=(2, 2),\n"
        "    n_latent_active_range=(1, 1),\n"
        "    edge_budget={'cy': (3, 3), 'zy': (2, 2), 'dc': (3, 3), 'dy': (1, 1),\n"
        "                 'dz': (0, 0), 'zc': (0, 0), 'cc': (0, 0), 'zz': (0, 0)},\n"
        "    dc_coeff_range=(0.6, 1.2),   # turn the confounding up so it is visible\n"
        "    dy_coeff_range=(0.4, 0.8),\n"
        "    n_cells=2, draws_per_cell=1, seed=20260728,\n"
        "    **pin('geometric', 'michaelis_menten'),\n"
        ")\n"
        "confounded = pg.sample_scm(confounded_cfg, seed=2, name='confounded')\n"
        "show(viz.plot_dag, confounded, 'D drives spend AND sales: a hidden confounder')"
    ),
    code(
        "start = confounded.cfg.l_max\n"
        "demand = np.asarray(confounded.data['demand'])[start:, 0]\n"
        "spend = np.asarray(confounded.data['channels'])[start:]\n"
        "base = np.asarray(confounded.data['channels_base'])[start:]   # spend with D -> C cut\n"
        "baseline = np.asarray(confounded.data['baseline'])[start:]\n"
        "\n"
        "print('D -> C loadings (w_dc)    :', np.round(np.asarray(confounded.params['w_dc']), 3))\n"
        "print('D -> Y loading  (delta_dy):', np.round(np.asarray(confounded.params['delta_dy']), 3))\n"
        "print()\n"
        "print('corr(D, baseline)                :',\n"
        "      round(float(np.corrcoef(demand, baseline)[0, 1]), 3))\n"
        "print('corr(D, D-driven part of spend)  :',\n"
        "      np.round([np.corrcoef(demand, (spend - base)[:, k])[0, 1]\n"
        "                for k in range(confounded.n_treatments)], 3))\n"
        "print('share of each channel sd from D  :',\n"
        "      np.round((spend - base).std(0) / spend.std(0), 3))\n"
        "print()\n"
        "print('corr(D, REALISED spend)          :',\n"
        "      np.round([np.corrcoef(demand, spend[:, k])[0, 1]\n"
        "                for k in range(confounded.n_treatments)], 3))"
    ),
    md(
        "Read those last two blocks together — the lesson is why you cannot detect\n"
        "confounding from a correlation.\n"
        "\n"
        "`channels_base` is the counterfactual spend path with every incoming arrow\n"
        "cut, so `channels - channels_base` is *exactly* the part of spend that `D`\n"
        "caused. Its correlation with `D` is ~1.0 and it carries a large share of each\n"
        "channel's variation: the confounding is structurally strong and unambiguous.\n"
        "\n"
        "Yet `corr(D, realised spend)` is all over the place and can even come out\n"
        "**negative**, because the channel's own independent walk dominates its level\n"
        "and two smooth series of 100 weeks have a large sample correlation by chance.\n"
        "A naive regression of sales on spend absorbs the `D`-driven part into `beta`,\n"
        "and no diagnostic on the observables reveals it."
    ),
    code(
        "show(viz.plot_timeseries, confounded, 'Confounded world: spend, controls, latent demand, sales')"
    ),
    md(
        "A second lever is available for the same story without an extra node:\n"
        "`confounding_strength_range` draws a per-world `rho` that mixes the baseline\n"
        "innovation into every channel innovation. That makes the model\n"
        "*semi-Markovian* — an unobserved common cause with no node of its own — so\n"
        "back-door adjustment on observables is not licensed either. Every\n"
        '"the graph satisfies the adjustment criterion" claim in these docs assumes\n'
        "`rho = 0`."
    ),
    code(
        "rho_cfg = pg.make_scm_prior(\n"
        "    n_treatments=2, n_covariates=1, n_latent=1, n_time_steps=104,\n"
        "    n_treatments_active_range=(2, 2), n_covariates_active_range=(1, 1),\n"
        "    n_latent_active_range=(1, 1),\n"
        "    edge_budget={'cy': (2, 2), 'zy': (1, 1), 'dy': (0, 0), 'dc': (0, 0),\n"
        "                 'dz': (0, 0), 'zc': (0, 0), 'cc': (0, 0), 'zz': (0, 0)},\n"
        "    confounding_strength_range=(0.6, 0.9),\n"
        "    n_cells=2, draws_per_cell=1, seed=20260728,\n"
        "    **pin('geometric', 'michaelis_menten'),\n"
        ")\n"
        "rho_world = pg.sample_scm(rho_cfg, seed=3, name='rho')\n"
        "print('drawn confounding strength rho =',\n"
        "      f\"{float(rho_world.data['confounding_strength']):.3f}\")\n"
        "print('graph has no D edges:',\n"
        "      not any(np.asarray(rho_world.g[k]).any() for k in ('g_dc', 'g_dy', 'g_dz')))"
    ),
    # ---------------------------------------------------------------- 9
    md(
        "## 9 · A corpus of many worlds\n"
        "\n"
        "`sample_prior_predictive` draws many worlds at once and returns flat numpy\n"
        "arrays — a corpus. Inactive slots are zero-padded and masked, so a consumer\n"
        "sees one fixed layout whether a task's `n_treatments_active` is 2 or 8.\n"
        "\n"
        "Widening the `*_active_range` arguments and dropping the `edge_budget` pins\n"
        "gives graph-size and structure variety across tasks."
    ),
    code(
        "corpus_cfg = pg.make_scm_prior(\n"
        "    n_treatments=5, n_covariates=3, n_latent=2, n_time_steps=104,\n"
        "    n_treatments_active_range=(2, 5),      # graph size varies per task\n"
        "    n_covariates_active_range=(1, 3),\n"
        "    n_latent_active_range=(1, 2),\n"
        "    edge_budget={'cy': (2, 5), 'zy': (1, 3), 'dc': (0, 4), 'dy': (0, 2),\n"
        "                 'zc': (0, 3), 'cc': (0, 2), 'dz': (0, 2), 'zz': (0, 1)},\n"
        "    n_cells=6, draws_per_cell=2, seed=20260728,\n"
        ")\n"
        "corpus = pg.sample_prior_predictive(corpus_cfg)\n"
        "print('tasks:', len(corpus['sales_raw']),\n"
        "      '| elapsed:', f\"{corpus['diagnostics']['timing']['elapsed_s']:.1f}s\")\n"
        "print('\\nkey                          shape')\n"
        "for key in ('spend_raw', 'controls', 'sales_raw', 'contributions_raw',\n"
        "            'treatment_active_mask', 'g'):\n"
        "    print(f'{key:28s} {np.asarray(corpus[key]).shape}')"
    ),
    code(
        "errors = pg.DataGenerator.validate_corpus(corpus)\n"
        "print('validation errors:', errors or 'none')\n"
        "assert errors == []\n"
        "\n"
        "print('\\nactive counts per task (treatments / covariates / latent):')\n"
        "for i in range(len(corpus['n_treatments_active'])):\n"
        "    print(f\"  task {i:2d}  {corpus['n_treatments_active'][i]} / \"\n"
        "          f\"{corpus['n_covariates_active'][i]} / {corpus['n_latent_active'][i]}\")"
    ),
    md(
        "Every corpus carries a self-describing contract in\n"
        "`diagnostics['signal']`, including the outcome-noise semantics a consumer\n"
        "must agree with before scoring anything."
    ),
    code(
        "signal = corpus['diagnostics']['signal']\n"
        "print('outcome-noise contract:', signal['outcome_noise_semantics'],\n"
        "      'v' + str(signal['outcome_noise_version']),\n"
        "      '| scale mode:', signal['outcome_std_mode'])\n"
        "print('adstock kernel       :', signal['adstock_kernel_semantics'],\n"
        "      'v' + str(signal['adstock_kernel_version']))\n"
        "print('metric layout        :', signal['metric_layout'])"
    ),
    md(
        "Signal diagnostics also report how much of each per-channel target actually\n"
        "carries information, and `check_signal_gate` turns them into a pass/fail.\n"
        "\n"
        "The gated quantities are **fractions over direct channels**, so they need a\n"
        "few hundred of them to be stable — the 12-task corpus above has far too few.\n"
        "Draw a wider, shorter corpus for the gate."
    ),
    code(
        "from prior_generator.signal_diagnostics import check_signal_gate\n"
        "\n"
        "gate_cfg = pg.make_scm_prior(\n"
        "    n_treatments=6, n_covariates=3, n_latent=2, n_time_steps=52,   # shorter, wider\n"
        "    n_treatments_active_range=(3, 6), n_covariates_active_range=(1, 3),\n"
        "    n_latent_active_range=(1, 2),\n"
        "    n_cells=10, draws_per_cell=5, seed=20260728,\n"
        ")\n"
        "gate_signal = pg.sample_prior_predictive(gate_cfg)['diagnostics']['signal']\n"
        "print('direct channels measured:', gate_signal['n_direct_channels'])\n"
        "passed, lines = check_signal_gate(gate_signal)\n"
        "for line in lines:\n"
        "    print(' ', line)\n"
        "print('\\ngate passes:', passed)"
    ),
    # ---------------------------------------------------------------- 10
    md(
        "## 10 · How large is everything?\n"
        "\n"
        "Everything above describes a corpus in **parameter** space (which arrows,\n"
        "which coefficients) or **signal** space (is the target textured enough to\n"
        "learn). Neither answers the magnitude question you ask of a prior before\n"
        "trusting it: *how large are the outcomes, and how large are the pieces that\n"
        "add up to them?*\n"
        "\n"
        "`outcome_distributions` pools every world along the **quantity** axis. Each\n"
        "row below is one quantity over all worlds: `units` counts the pooled units,\n"
        "`zero` is the fraction of units that are identically zero, and the\n"
        "quantiles describe the distribution."
    ),
    code("dist = pg.outcome_distributions(corpus)\nprint(dist.table())"),
    md(
        "A **unit** is one world for a scalar quantity (`sales`, `baseline`, ...)\n"
        "and one `(world, channel)` / `(world, control)` / `(world, latent)` pair for\n"
        "a column quantity. Padded inactive columns are dropped, so no zero padding\n"
        "reaches a statistic.\n"
        "\n"
        "The same object in **share** units answers the attribution-size question:\n"
        "`unit_share = Σ_t value / Σ_t sales`. Because the decomposition is exact,\n"
        "these shares are a real budget — the additive pieces sum to 1.0 per world."
    ),
    code(
        "print(dist.table(of='share'))\n"
        "\n"
        "budget = dist.additive_share_total()\n"
        "print(f'\\nadditive shares sum to {budget.min():.6f} .. {budget.max():.6f} '\n"
        "      'per world (exact decomposition)')"
    ),
    md(
        "Structurally-null channels — active, spend observed, no `C→Y` arrow — stay\n"
        "in as exact zeros rather than being silently filtered, because they are a\n"
        "real outcome of the prior and they do drag every contribution statistic\n"
        "toward zero. `zero_unit_fraction` reports them; `select` conditions on\n"
        "units when you want direct channels only."
    ),
    code(
        "media = dist['channel_contribution']\n"
        "direct = media.select(media.unit_max > 0)\n"
        "print(f'active channels        : {media.n_units}'\n"
        "      f' ({media.zero_unit_fraction:.0%} have no C->Y arrow)')\n"
        'print(f\'median share, all      : {media.quantiles(of="share")["q50"]:.3f}\')\n'
        'print(f\'median share, direct   : {direct.quantiles(of="share")["q50"]:.3f}\')\n'
        "\n"
        "print('\\nper-world media share (Σ direct + indirect) / Σ sales:')\n"
        "total_media = dist['media_contribution']\n"
        "for level in (0.05, 0.5, 0.95):\n"
        "    q = np.quantile(total_media.unit_share, level)\n"
        "    print(f'  q{level * 100:>4.0f}: {q:.3f}')"
    ),
    md(
        "Conditioning is a world row mask, not a separate API — anything you can\n"
        "express over corpus rows works, e.g. one cell of the corpus at a time."
    ),
    code(
        "for cell in np.unique(corpus['cell_id'])[:3]:\n"
        "    d = pg.outcome_distributions(corpus, worlds=corpus['cell_id'] == cell)\n"
        '    print(f"cell {cell}: {d.n_worlds} worlds"\n'
        "          f\" | median Y {d['sales'].pooled['q50']:7.2f}\"\n"
        "          f\" | media share {np.median(d['media_contribution'].unit_share):.3f}\")"
    ),
    md(
        "Worlds have arbitrary sales levels, so pooled *raw* values mix scales.\n"
        "`normalize='sales_scale'` divides every Y-scale quantity by the world's own\n"
        "scale; shares are ratios and never move. The histogram grid shows the\n"
        "whole set of worlds at once — spread in outcome space, not parameter space."
    ),
    code(
        "fig_path = os.path.join(tempfile.mkdtemp(), 'outcomes.png')\n"
        "viz.plot_outcome_distributions(dist, fig_path, of='share')\n"
        "display(Image(filename=fig_path))"
    ),
    md(
        "`dist.summary()` is JSON-ready for a report, and `dist.to_frame()` is a\n"
        "long-form pandas table with one row per unit."
    ),
    code(
        "frame = dist.to_frame()\n"
        "print(frame.groupby('quantity')[['mean', 'std', 'share']]\n"
        "      .median(numeric_only=True).round(4).to_string())"
    ),
    md(
        "## 11 · Persist a corpus and an auditable bundle\n"
        "\n"
        "`save_corpus` / `load_corpus` round-trip a compressed `.npz` that needs only\n"
        "numpy to read. `write_scm_bundle` writes one world as a human-auditable\n"
        "folder: CSVs, the graph, the equations, and the realised parameters."
    ),
    code(
        "root = tempfile.mkdtemp()\n"
        "npz = os.path.join(root, 'corpus.npz')\n"
        "pg.save_corpus(corpus, npz)\n"
        "loaded = pg.load_corpus(npz)\n"
        "print('corpus round-trips exactly:',\n"
        "      bool(np.array_equal(corpus['spend_raw'], loaded['spend_raw'])))\n"
        "\n"
        "bundle = pg.write_scm_bundle(confounded, os.path.join(root, 'world'))\n"
        "print('bundle files:', sorted(os.listdir(bundle)))"
    ),
    md(
        "## Recap\n"
        "\n"
        "You went from a one-arrow graph to a confounded five-channel corpus, one\n"
        "parameter group at a time:\n"
        "\n"
        "1. `edge_budget` places the arrows; the three sizes pin the layout.\n"
        "2. Observables are `channels`, `controls`, `sales`. Everything else is truth.\n"
        "3. The decomposition is exact — `identity_error()` proves it per world.\n"
        "4. Media response is `beta · saturation(adstock(spend))`, κ-relative.\n"
        "5. Channel texture is what makes the response *identifiable*.\n"
        "6. `rw_baseline_std_range` / `rw_sales_std_range` set the residual/media\n"
        "   ratio — the axis that decides whether any model can recover a world.\n"
        "7. `zc` and `cc` add observed confounding and channel interaction.\n"
        "8. `dc` + `dy` (or `confounding_strength_range`) break causal sufficiency.\n"
        "9. `outcome_distributions` reads the corpus in outcome space: how large Y\n"
        "   gets, and what share of it each cause accounts for.\n"
        "\n"
        "Next: [MMM recovery case study](simple-model.ipynb) fits a real MMM to one\n"
        "of these worlds, shows where it fails, and compares it against the posterior\n"
        "oracle. The [API reference](../reference/index.md) documents every public\n"
        "symbol."
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
    "index.ipynb",
)
with open(out, "w") as fh:
    nbf.write(nb, fh)
print("wrote", out, f"({len(cells)} cells)")
