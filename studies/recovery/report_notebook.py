"""Pure notebook template generator: saved evidence only, no stage execution."""

from __future__ import annotations

import os
from pathlib import Path

import nbformat as nbf


def notebook(out: Path) -> dict:
    project = Path(__file__).resolve().parents[2]
    relative = os.path.relpath(project, out)
    cells = [
        nbf.v4.new_markdown_cell(
            "# Contribution recovery · run-bound evidence\n\n"
            "**Run All reads saved, hashed artifact inputs and renders only.** It cannot generate or fit. "
            "Legacy records are manifest/input-hash compatible only; historical smoke receipts "
            "are verified under their *recorded* identities, never recertified as current fits. "
            "All scientific conclusions remain **pending review for this exact run**."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport sys\n"
            f"sys.path.insert(0, str((Path.cwd() / {relative!r}).resolve()))\n"
            "from studies.recovery.reporting import read_report\n"
            "from IPython.display import display, Markdown\n"
            "import pandas as pd\nimport numpy as np\n"
            "import plotly.graph_objects as go\nfrom plotly.subplots import make_subplots\n"
            "import ipywidgets as widgets\n"
            "evidence = read_report(Path.cwd())\n"
            'display(Markdown(f\'**Evidence ID:** `{evidence["run_id"]}` · **Source:** `{evidence["kind"]}` · \'\n'
            "                 f'**Provenance:** {evidence[\"provenance\"]}'))"
        ),
        nbf.v4.new_markdown_cell(
            "## Configuration, provenance and coverage\n\n"
            "The evidence ID hashes the resolved config, input files, report sources and environment. "
            "Changed inputs fail Run All. Eligible channels exclude exact known-zero effects; "
            "a zero denominator means **not represented**, not zero failures. "
            "Small groups (<10 evaluable channels) are flagged."
        ),
        nbf.v4.new_code_cell(
            "display(Markdown('**Resolved configuration:**'))\n"
            "display(pd.json_normalize(evidence['resolved_config']).T.rename(columns={0: 'resolved value'}))\n"
            "display(pd.DataFrame([{'method': k, 'worlds': len({r['world_id'] for r in evidence['worlds'] "
            "if r.get('method', evidence['kind']) == k}), 'channels': sum(1 for c in evidence['channels'] "
            "if c.get('method', evidence['kind']) == k), 'eligible': sum(c['nonzero_effect'] for c in evidence['channels'] "
            "if c.get('method', evidence['kind']) == k)} for k in evidence['pattern_rates']]))\n"
            "display(Markdown('**Optional exploration:** ' + str(evidence['capabilities'])))"
        ),
        nbf.v4.new_markdown_cell(
            "## Noise, true parameters and variation\n\n"
            "These are **oracle descriptive profiles**, not fit initialization and not predictors "
            "of recovery. Noise SD, baseline variation, media/control contribution SD, "
            "carryover alpha, MM multiplier and direct coefficients come from saved worlds."
        ),
        nbf.v4.new_code_cell(
            "profiles = pd.json_normalize([{'world': r['world_id'], 'method': r.get('method', evidence['kind']), "
            "**r['profile']} for r in evidence['worlds']])\n"
            "display(profiles[['world', 'method', 'realized_noise_sd', 'baseline_temporal_sd', "
            "'carryover_alpha', 'mm_kappa_mult', 'media_beta', 'control_coefficient', "
            "'media_true_temporal_sd', 'control_true_temporal_sd']].head(60))"
        ),
        nbf.v4.new_markdown_cell(
            "## Computational diagnostics ≠ point recovery ≠ uncertainty\n\n"
            "Invalid MCMC exploration is unresolved, *not* a point failure. Short-smoke "
            "ADVI finite optimization only is **not** a convergence certificate; R-hat/ESS "
            "do not apply to variational draws. OLS has no posterior uncertainty. "
            "Pointwise credible intervals describe saved posterior uncertainty, not "
            "causal or frequentist generalization to new datasets."
        ),
        nbf.v4.new_code_cell(
            "diagnostics = pd.json_normalize([{'world': r['world_id'], "
            "'method': r.get('method', evidence['kind']), 'status': r['computational_status'], "
            "**r['diagnostics']} for r in evidence['worlds']])\n"
            "display(diagnostics.head(60))\n"
            "recovery = pd.DataFrame([{'world': r['world_id'], "
            "'method': r.get('method', evidence['kind']), 'status': r['computational_status'], "
            "'media': r['recovery']['media_pass'], 'controls': r['recovery']['control_pass'], "
            "'joint': r['recovery']['joint_pass'], 'total_media': (r['recovery']['total_media'] or {}).get('pass'), "
            "'baseline': (r['recovery']['baseline'] or {}).get('pass'), "
            "'uncertainty': r['uncertainty']['type']} for r in evidence['worlds']])\n"
            "display(recovery.head(60))"
        ),
        nbf.v4.new_markdown_cell(
            "## Pattern-specific **channel** failure rates\n\n"
            "Rows count **the channels bearing the named pattern**, not all failures in "
            "the same world. Patterns overlap and upward/downward trends have distinct "
            "denominators. Total includes known-zero channels; eligible excludes known-zero; "
            "unresolved eligible channels do not enter the failure-rate denominator."
        ),
        nbf.v4.new_code_cell(
            "for method, rates in evidence['pattern_rates'].items():\n"
            "    display(Markdown('### ' + method))\n"
            "    display(pd.DataFrame(rates)[['kind','pattern','total','eligible','known_zero',"
            "'evaluable','failed','unresolved','failure_rate','status','small_count_warning']])\n"
            "channel_table = pd.DataFrame(evidence['channels'])\n"
            "display(channel_table.head(25))"
        ),
        nbf.v4.new_markdown_cell(
            "## Selectable true/fitted trajectories\n\n"
            "Select a world/method and a **nonzero** contribution. Separate Plotly panels "
            "show its contribution (and available 95% interval), raw media spend for media "
            "channels, and the baseline (always retained). Legends sit below the panels. "
            "Known-zero padded/null channels are hidden; absence is not a success."
        ),
        nbf.v4.new_code_cell(
            "keys = sorted(evidence['trajectories'])\n"
            "selection = widgets.Dropdown(options=keys, value=keys[0], description='World · method:', layout=widgets.Layout(width='450px'))\n"
            "def _eligible(key):\n"
            "    if key not in evidence['trajectories']:\n"
            "        return ('Baseline',)\n"
            "    t = evidence['trajectories'][key]\n"
            "    truth = np.asarray(t['truth'])\n"
            "    return tuple(name for i, name in enumerate(t['terms']) if i == 0 or np.linalg.norm(truth[:, i]) > 0)\n"
            "initial = _eligible(selection.value)\n"
            "channel = widgets.Dropdown(options=initial, value=initial[0], description='Channel:', layout=widgets.Layout(width='300px'))\n"
            "def _update_channels(change=None):\n"
            "    valid = _eligible(selection.value)\n"
            "    wanted = channel.value if channel.value in valid else valid[0]\n"
            "    channel.options = valid\n"
            "    channel.value = wanted\n"
            "selection.observe(_update_channels, names='value')\n"
            "def _view(key, name):\n"
            "    if key not in evidence['trajectories'] or name not in _eligible(key):\n"
            "        return None  # transient selector update, not a report/plot error\n"
            "    t = evidence['trajectories'][key]\n"
            "    ix = t['terms'].index(name)\n"
            "    week = t['weeks']\n"
            "    truth, fit = np.asarray(t['truth']), np.asarray(t['fitted'])\n"
            "    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, "
            "subplot_titles=(name + ' contribution', 'Raw media spend (if media)', 'Baseline contribution'))\n"
            "    def line(row, values, label, dash='solid'):\n"
            "        fig.add_trace(go.Scatter(x=week, y=values, name=label, mode='lines', "
            "line={'dash': dash}, legendgroup=label, showlegend=True), row=row, col=1)\n"
            "    line(1, truth[:, ix], 'True ' + name)\n    line(1, fit[:, ix], 'Fitted ' + name, 'dash')\n"
            "    if t['lower95'] is not None:\n"
            "        low, high = np.asarray(t['lower95'])[:, ix], np.asarray(t['upper95'])[:, ix]\n"
            "        fig.add_trace(go.Scatter(x=week, y=low, line={'width':0}, showlegend=False, hoverinfo='skip'), row=1, col=1)\n"
            "        fig.add_trace(go.Scatter(x=week, y=high, fill='tonexty', fillcolor='rgba(80,80,180,.17)', "
            "line={'width':0}, name='95% pointwise interval'), row=1, col=1)\n"
            "    if name.startswith('C'):\n"
            "        line(2, np.asarray(t['raw_spend'])[:, int(name[1:])-1], 'Raw spend ' + name)\n"
            "    line(3, truth[:, 0], 'True baseline')\n    line(3, fit[:, 0], 'Fitted baseline', 'dash')\n"
            "    status = next((r['computational_status'] for r in evidence['worlds'] if "
            "str(r['world_id']) + (':' + r['method'] if 'method' in r else '') == key), 'unknown')\n"
            "    fig.update_layout(height=800, template='plotly_white', legend={'orientation':'h', 'y':-.13, 'x':0}, "
            "title=key + ' · ' + name + ' · ' + status)\n    display(fig)\n    return fig\n"
            "display(widgets.HBox([selection, channel]))\n"
            "interactive = widgets.interactive_output(_view, {'key': selection, 'name': channel})\n"
            "display(interactive)"
        ),
        nbf.v4.new_markdown_cell(
            "## Interpretation · run-bound review\n\n"
            "Pending until an explicit reviewer approves this exact run into a separate sealed "
            "sibling review artifact. A present but invalid/cross-run review FAILS CLOSED. "
            "Optional held-out-world logistic screening, t-SNE and correlated-pair analysis "
            "are **deferred/unsupported**."
        ),
        nbf.v4.new_code_cell(
            "from studies.recovery.review_interpretation import attached_review\n"
            "review = attached_review(Path.cwd())\n"
            "if review is None:\n"
            "    display(Markdown(Path('interpretation.md').read_text()))\n"
            "else:\n"
            "    display(Markdown('**Reviewed for run** `' + review['report_run_id'] + "
            "'` by **' + review['reviewer'] + '** · decision: approved · review `' + review['review_id'] + '`'))\n"
            "    display(Markdown(review['interpretation']))"
        ),
    ]
    nb = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Recovery study isolated Python",
                "language": "python",
                "name": "recovery-study",
            },
            "language_info": {"name": "python"},
        },
    )
    return dict(nb)
