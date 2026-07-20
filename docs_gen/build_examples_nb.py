"""Assemble ``docs/examples/index.ipynb`` from ordered cells with nbformat.

Run from the repo root:  python docs_gen/build_examples_nb.py

The notebook is self-contained (only imports ``prior_generator``) so a reader can
download it and run it as-is. It is executed at docs-build time by mkdocs-jupyter.
"""

from __future__ import annotations

import os

import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md(
        "# Examples — an end-to-end tour\n"
        "\n"
        "This notebook samples synthetic marketing worlds, inspects everything "
        "inside them, plots several time series, and **validates** that the "
        "reported decomposition is exact. It only needs `prior_generator`, so you "
        "can [download it](index.ipynb) and run it yourself.\n"
        "\n"
        "> Every cell below is executed when the docs are built — the outputs and "
        "figures you see are real."
    ),
    code(
        "%matplotlib inline\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')          # hide harmless tqdm/ipywidgets notices\n"
        "\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from IPython.display import Image, display\n"
        "\n"
        "import prior_generator as pg\n"
        "from prior_generator import viz          # the plotting submodule\n"
        "print('prior-generator', pg.__version__)"
    ),
    md(
        "## 1 · Sample a world\n"
        "\n"
        "We draw from the `confounded_spend` scenario — latent demand drives both "
        "spend and the baseline, the classic MMM confounding. Sampling is "
        "deterministic in `(cfg, seed)`."
    ),
    code(
        "sc = pg.SCENARIOS[1]                       # confounded_spend\n"
        "scm = pg.sample_scm(sc.prior(T=104, seed=0), seed=0,\n"
        "                    connect_all=sc.connect_all,\n"
        "                    name=sc.name, purpose=sc.purpose)\n"
        "\n"
        "print(f'world {scm.name!r}: K={scm.K} channels, M={scm.M} controls, '\n"
        "      f'J={scm.J} demand, T={scm.T} weeks')"
    ),
    md(
        "## 2 · Access everything inside the `SCM`\n"
        "\n"
        "`.data` holds the 13 named series, `.g` the DAG blocks, `.params` the "
        "drawn coefficients and mechanisms."
    ),
    code(
        "print('data series:')\n"
        "for name, arr in scm.data.items():\n"
        "    print(f'  {name:<28} {arr.shape}')\n"
        "\n"
        "print('\\nDAG blocks (arrow counts):')\n"
        "for name, block in scm.g.items():\n"
        "    print(f'  {name:<7} {int(np.asarray(block).sum())}')"
    ),
    code(
        "# Per-channel mechanisms and the direct-response coefficients\n"
        "from prior_generator.worlds import channel_role, mechanism_label\n"
        "for k in range(scm.K):\n"
        "    print(f'C{k+1}: role={channel_role(scm.g, k):<7} '\n"
        "          f'mechanism={mechanism_label(scm.params, k):<20} '\n"
        "          f'beta={scm.params[\"beta\"][k]:+.2f}')"
    ),
    md(
        "## 3 · Read the world's story\n"
        "\n"
        "`describe_scm` renders the DAG with coefficients, node connectivity, "
        "mechanisms, the decomposition-identity check, and signal metrics as plain "
        "text."
    ),
    code("print(pg.describe_scm(scm))"),
    md(
        "## 4 · The causal graph\n"
        "\n"
        "The package renders four figures. They write to a path, so we render to a "
        "temp file and display it inline."
    ),
    code(
        "import tempfile, os\n"
        "def show_fig(plot_fn, world, title=None):\n"
        "    path = os.path.join(tempfile.mkdtemp(), 'fig.png')\n"
        "    plot_fn(world, path, title or world.name)\n"
        "    display(Image(filename=path))\n"
        "\n"
        "show_fig(viz.plot_dag, scm)"
    ),
    md(
        "## 5 · The observable time series\n"
        "\n"
        "What a modeller actually sees: per-channel spend, the observed controls "
        "and (latent) demand, and sales."
    ),
    code("show_fig(viz.plot_timeseries, scm)"),
    md(
        "### Plot the series yourself\n"
        "\n"
        "`.data` is plain numpy, so you can plot any slice directly. Here is each "
        "channel's spend on its own axes."
    ),
    code(
        "weeks = np.arange(scm.T)\n"
        "fig, ax = plt.subplots(figsize=(10, 4))\n"
        "for k in range(scm.K):\n"
        "    ax.plot(weeks, scm.data['channels'][:, k], lw=1.4, label=f'C{k+1}')\n"
        "ax.set(title='Media spend by channel', xlabel='week', ylabel='spend')\n"
        "ax.legend(ncol=scm.K, frameon=False)\n"
        "ax.spines[['top', 'right']].set_visible(False)\n"
        "plt.show()"
    ),
    md(
        "## 6 · Per-channel spend vs its true contribution\n"
        "\n"
        "Because we simulated the world, we know each channel's *true* "
        "contribution to sales — not an estimate. Indexed to mean 1, you can see "
        "how spend sweeps its (adstocked, saturated) response."
    ),
    code("show_fig(viz.plot_channels, scm)"),
    md(
        "## 7 · The exact decomposition\n"
        "\n"
        "Sales equals the sum of its true components to float precision. Let's "
        "prove it, then visualise every piece."
    ),
    code(
        "d = scm.data\n"
        "lhs = d['sales']\n"
        "rhs = d['baseline'] + d['contributions'].sum(1) + d['indirect_effects']\n"
        "print('sales = baseline + Σ direct + indirect')\n"
        "print('  max |error| =', f'{np.abs(lhs - rhs).max():.2e}')\n"
        "\n"
        "# the fully-unrolled identity == SCM.reconstruction()\n"
        "print('reconstruction() == sales, max err =', f'{scm.identity_error():.2e}')"
    ),
    code("show_fig(viz.plot_decomposition, scm)"),
    md(
        "### A stacked view of where sales come from\n"
        "\n"
        "The average sales dollar, split into baseline, direct media, and the three "
        "indirect sources — a compact custom time-series view built from `.data`."
    ),
    code(
        "src = d['indirect_effects_by_source']       # (T, 3): cc, zc, dc\n"
        "layers = {\n"
        "    'baseline': d['baseline'],\n"
        "    'direct media': d['contributions'].sum(1),\n"
        "    'indirect · cc': src[:, 0],\n"
        "    'indirect · zc': src[:, 1],\n"
        "    'indirect · dc': src[:, 2],\n"
        "}\n"
        "fig, ax = plt.subplots(figsize=(10, 4))\n"
        "ax.stackplot(weeks, *layers.values(), labels=list(layers), alpha=0.9)\n"
        "ax.plot(weeks, d['sales'], color='k', lw=1.2, label='sales Y')\n"
        "ax.set(title='Sales, decomposed', xlabel='week', ylabel='sales')\n"
        "ax.legend(loc='upper left', ncol=3, frameon=False, fontsize=8)\n"
        "ax.spines[['top', 'right']].set_visible(False)\n"
        "plt.show()"
    ),
    md(
        "## 8 · Different worlds, different time series\n"
        "\n"
        "Each scenario isolates a different causal pathway. Overlaying their sales "
        "series (indexed to mean 1) shows how varied the generated dynamics are."
    ),
    code(
        "fig, axes = plt.subplots(len(pg.SCENARIOS), 1, figsize=(10, 9), sharex=True)\n"
        "for idx, ax in enumerate(axes):\n"
        "    sc = pg.SCENARIOS[idx]\n"
        "    w = pg.sample_scm(sc.prior(T=104, seed=0), seed=0,\n"
        "                      connect_all=sc.connect_all, name=sc.name)\n"
        "    y = w.data['sales']\n"
        "    ax.plot(np.arange(w.T), y / y.mean(), lw=1.3)\n"
        "    ax.set_ylabel(f'{idx}: {sc.name}', fontsize=8, rotation=0, ha='right', va='center')\n"
        "    ax.spines[['top', 'right']].set_visible(False)\n"
        "axes[-1].set_xlabel('week')\n"
        "fig.suptitle('Sales across the five scenarios (indexed to mean 1)')\n"
        "fig.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "## 9 · Generate and validate a corpus\n"
        "\n"
        "Scale up to a corpus — a dict of stacked arrays over many worlds. "
        "`DataGenerator.validate_corpus` checks the schema; the additive identity "
        "holds across every task."
    ),
    code(
        "cfg = pg.make_scm_prior(n_treatments=5, n_covariates=3, n_latent=2,\n"
        "                        edge_budget={'cy': (4, 4), 'dc': (2, 2), 'zc': (1, 2)},\n"
        "                        n_cells=2, draws_per_cell=2, seed=42)\n"
        "corpus = pg.sample_prior_predictive(cfg)\n"
        "print('N tasks:', corpus['spend_raw'].shape[0])\n"
        "\n"
        "from prior_generator.data_generator import DataGenerator\n"
        "print('validation errors:', DataGenerator.validate_corpus(corpus) or 'none — OK')"
    ),
    code(
        "lhs = corpus['sales_raw'].astype(np.float64)\n"
        "rhs = (corpus['baseline_raw'] + corpus['contributions_raw'].sum(-1)\n"
        "       + corpus['indirect_effects']).astype(np.float64)\n"
        "print('additive identity across the whole corpus:')\n"
        "print('  max abs error =', f'{np.abs(lhs - rhs).max():.2e}')"
    ),
    md(
        "### Signal diagnostics\n"
        "\n"
        "Every corpus embeds a signal summary so weak-signal priors are caught at "
        "generation time. (This demo corpus is tiny, so a gate row may read FAIL "
        "purely from small-sample noise.)"
    ),
    code(
        "from prior_generator.signal_diagnostics import check_signal_gate\n"
        "ok, lines = check_signal_gate(corpus['diagnostics']['signal'])\n"
        "for line in lines:\n"
        "    print(line)"
    ),
    md(
        "## 10 · Persist a corpus and a bundle\n"
        "\n"
        "`save_corpus` / `load_corpus` round-trip a compressed `.npz`; "
        "`write_scm_bundle` writes a human-auditable folder."
    ),
    code(
        "import tempfile, os\n"
        "root = tempfile.mkdtemp()\n"
        "\n"
        "npz = os.path.join(root, 'corpus.npz')\n"
        "pg.save_corpus(corpus, npz)\n"
        "loaded = pg.load_corpus(npz)\n"
        "print('corpus round-trips exactly:',\n"
        "      bool(np.array_equal(corpus['spend_raw'], loaded['spend_raw'])))\n"
        "\n"
        "bundle = pg.write_scm_bundle(scm, os.path.join(root, 'world'))\n"
        "print('bundle files:', sorted(os.listdir(bundle)))"
    ),
    md(
        "## Recap\n"
        "\n"
        "From one seed you drew a world, reached every observable and every truth "
        "series inside it, plotted the graph and several time series, **proved** the "
        "decomposition is exact, then scaled up to a validated corpus and persisted "
        "both a corpus and an auditable bundle.\n"
        "\n"
        "Next: the [API reference](reference/index.md) documents every public "
        "symbol, generated from the source."
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
