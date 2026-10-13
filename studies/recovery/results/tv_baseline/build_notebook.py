"""Rebuild + execute the committed STATIC report using only this compact directory.

No generation, fitting, inference, old archive, optional smoothing or review action.
Requires numpy/pandas/matplotlib/nbformat/nbclient/ipykernel and a Python kernelspec.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


def notebook() -> nbf.NotebookNode:
    md = nbf.v4.new_markdown_cell
    py = nbf.v4.new_code_cell
    return nbf.v4.new_notebook(
        cells=[
            md("""# Historical 50-world recovery studies · compact result analysis

**Archived studies, not a new PR1 pipeline run.** Original source commit `4dc69ce`; historical approved report run IDs and receipt/checksum inventory are shown below. The approvals of Teemu Säilynoja (Plannotator `pn-2a768e`) cover the *old exact report identities*, not this notebook or a new fit. These two experiments use different baseline/outcome regimes and objectives: **do not rank TV-MCMC against bounded least squares**. No inference runs when this notebook executes. All visible figures are static PNG, and all tables are saved HTML outputs.

The TV baseline was applied as a matched baseline-only intervention on the constant corpus; it was not an independent draw from the PR1 varying example. Score only the 97 original weeks 7–103, using the 10%-of-true-channel norm or 1%-of-noiseless-outcome norm (whichever greater); a computationally unresolved fit is neither a recovery pass nor a recovery failure. Components, point gates, pattern groups and descriptors are recomputed from local compressed float64 contributions, spend and recorded status/metadata."""),
            py("""from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from analysis import LABELS, load_snapshot, summarize, trajectory

root = Path.cwd()
meta, arrays = load_snapshot(root)  # validates payload hashes, historical IDs, arrays and term/ledger inventory
report = summarize(meta, arrays)
print('Validated compact snapshot:', len(report['world_rows']), 'world-method rows;', len(report['channel_rows']), 'channel rows; 97 scored weeks')
"""),
            md("""## Historical provenance and approved narratives

Checksums below identify the *historical approved* reports and this compact copy. The old reports seal the evidence they read, but legacy source manifests contain **input hashes only**; missing original per-fit output and generator-source hashes cannot be retroactively certified. `historical-config.json` records the full original generator prior plus distinct matched-TV intervention, bounded optimizer and TV-NUTS settings. `provenance.json` contains the complete historical review/evidence/interpretation receipt hashes and every compact payload checksum. The byte-identical original interpretations are [`tv50.md`](tv50.md) and [`bounded50.md`](bounded50.md); their own headers name the older report IDs. No newly approved interpretation is claimed."""),
            py("""import json
provenance = json.loads((root / 'provenance.json').read_text())
display(pd.DataFrame([{'historical_regime': LABELS[method],
    'approved_old_run_id': record['historical_report_run_id'],
    'source_evidence_sha256': record['original_evidence_sha256'],
    'review_id': record['historical_review_id'],
    'interpretation_sha256': record['interpretation_sha256']}
    for method, record in provenance['historical_approvals'].items()]))
display(pd.DataFrame([{'compact_file': name, 'sha256': digest}
    for name, digest in provenance['compact_payload_sha256'].items()]))
"""),
            md("""## World-level recovery gates (not a cross-regime comparison)

Every individual nonzero media channel must pass for `media_pass`; `joint_pass` additionally requires every nonzero control. The total-media sum and intrinsic baseline are **separate** point estimands; all four world gates use the same historic 10% / 1% L2 allowance. The 2 bounded optimizer worlds 6 and 9 are unresolved, not attribution failures. An optimization stopping rule and an MCMC diagnostics pass are computational qualifications, *not* proofs of global optimum, posterior mode coverage or individual-parameter identifiability."""),
            py("""counts = pd.DataFrame(report['counts'])
display(counts[['regime','worlds','valid','unresolved','unresolved_ids',
                'media_pass','joint_pass','total_media_pass','baseline_pass']])
gate_names = [('media_pass','All individual media'),('joint_pass','All media and controls'),
              ('total_media_pass','Total-media trajectory'),('baseline_pass','Baseline trajectory')]
gates = pd.DataFrame([{'regime': c['regime'], 'estimand': text, 'passes': c[field],
    'evaluable_worlds': c['valid'], 'unresolved_worlds': c['unresolved']}
    for c in report['counts'] for field, text in gate_names])
display(gates)
fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout='constrained')
for axis, (_, group) in zip(axes, gates.groupby('regime', sort=False)):
    axis.barh(group['estimand'], group['passes'], color='#366b86')
    axis.set_xlim(0, 50); axis.set_xlabel('Established passes / evaluable worlds')
    axis.set_title(group['regime'].iloc[0] + f" · evaluable {group['evaluable_worlds'].iloc[0]}/50")
    for index, row in enumerate(group.itertuples()):
        axis.text(row.passes + .6, index, f'{row.passes}/{row.evaluable_worlds}', va='center', fontsize=8)
fig.suptitle('Separate estimands in each distinct regime — NOT a method ranking')
display(fig); plt.close(fig)
"""),
            md("""## Channel denominators and overlapping generated patterns

Worlds with more channels contribute more pooled rows; these are not independent experiments. A structurally known-zero channel is not eligible even if it bears a pattern. Eligible nonzero channels in a computationally unresolved world are recorded separately from evaluable failures. Pattern counts belong to the **channel carrying the assigned pattern**, not to another failure in that world. Patterns overlap and their rates do not add; an assigned pattern need not dominate the realized input."""),
            py("""display(pd.DataFrame([{'regime': c['regime'], 'kind': kind,
    'nonzero_eligible': c[kind+'_eligible'], 'evaluable': c[kind+'_evaluable'],
    'failed': c[kind+'_failed'], 'unresolved': c[kind+'_unresolved'],
    'known_zero': c[kind+'_known_zero'],
    'failure_rate_on_evaluable': c[kind+'_failed']/c[kind+'_evaluable']}
    for c in report['counts'] for kind in ('media','control')]))
for regime in ('tv','bounded'):
    rows = [r for r in report['pattern_rows'] if r['regime']==regime and r['kind']=='media']
    display(Markdown('### ' + LABELS[regime] + ' · media pattern-bearing channels'))
    display(pd.DataFrame(rows)[['pattern','total','eligible','known_zero','evaluable',
        'failed','unresolved','failure_rate','small_count_warning']])
"""),
            md("""## TV-MCMC: computational screen and descriptive contribution variation

Only the TV regime has posterior intervals (saved **pointwise** 95% intervals, not simultaneous trajectory or held-out calibration). True generated parameter/trajectory descriptors for passing and failing nonzero media channels are observational associations, not isolated causal effects or a generator tuning rule. Group sizes in the table and scatter legend are recomputed from the compact snapshot. Noise SD is the realized scored-window observation-noise SD; observed-sales mean is the denominator for the noise-fraction quartiles. CV = temporal SD / absolute temporal mean; medians include no rounding until display."""),
            py("""display(pd.DataFrame([report['tv_diagnostics']]))
display(pd.DataFrame(report['tv_descriptor_medians']))
display(pd.DataFrame([{'noise_SD / mean_observed_sales': label, 'TV_world_fraction': value}
    for label, value in zip(('25th percentile','median','75th percentile'),
                            report['tv_noise_fraction_quartiles'])]))
passed = [r for r in report['tv_descriptors'] if r['point_pass'] is True]
failed = [r for r in report['tv_descriptors'] if r['point_pass'] is False]
fig, axis = plt.subplots(figsize=(9, 5), layout='constrained')
for group, text, color in [(passed, 'TV channel passes', '#347b66'),
                           (failed, 'TV channel fails', '#b14f4d')]:
    axis.scatter([x['contribution_cv'] for x in group],
                 [x['contribution_sd_per_noise_sd'] for x in group],
                 s=19, alpha=.62, color=color, label=f'{text} ({len(group)})')
axis.set_xlabel('True contribution temporal CV (scored 97 weeks)')
axis.set_yscale('log')  # all eligible signal/noise ratios are positive; show small ones too
axis.set_ylabel('True contribution temporal SD / realized noise SD (log scale)')
axis.set_title('TV channel-level descriptors: descriptive, clustered by world')
axis.legend(); axis.grid(alpha=.2)
display(fig); plt.close(fig)
"""),
            py("""tv_patterns = [r for r in report['pattern_rows'] if r['regime']=='tv' and
               r['kind']=='media' and r['pattern']!='trend']  # directional trend categories instead of duplicate union
fig, axis = plt.subplots(figsize=(11, 5), layout='constrained')
axis.bar([r['pattern'] for r in tv_patterns], [r['failure_rate'] for r in tv_patterns], color='#547899')
for i, row in enumerate(tv_patterns):
    axis.text(i, row['failure_rate']+.012, f"{row['failed']}/{row['evaluable']}", ha='center', fontsize=8)
axis.set_ylim(0,.85); axis.tick_params(axis='x', labelrotation=35)
axis.set_ylabel('TV failures / evaluable nonzero pattern-bearing channels')
axis.set_title('TV media: overlapping pattern assignments · descriptive only')
axis.grid(axis='y',alpha=.2)
display(fig); plt.close(fig)
"""),
            md("""## Four **illustrative, preselected** paths · NOT representative cases

World **0**: TV channel **C2** fails its point gate; bounded channel **C3** also fails its *separately evaluated* point gate. World **12**: C6 passes its within-regime gate in each distinct baseline/outcome study. These channel IDs were chosen explicitly to show a valid failed and valid passing channel **within** each regime; they are not a matched algorithm comparison, typical-world estimate or automatic cherry-picked headline statistic. The plotted raw input is shared between regimes; the TV plot's shading is pointwise posterior intervals only. Figures display raw spend and the true/fitted baseline so differing baselines are not hidden."""),
            py("""for regime, world, channel, expected in [('tv',0,'C2',False),('tv',12,'C6',True),
                                          ('bounded',0,'C3',False),('bounded',12,'C6',True)]:
    rec = next(r for r in report['channel_rows'] if r['regime']==regime and
               r['world_id']==world and r['channel']==channel)
    assert rec['point_pass'] is expected
    path = trajectory(meta, arrays, regime, world)
    i = path['terms'].index(channel)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, layout='constrained')
    x = path['weeks']
    axes[0].plot(x,path['truth'][:,i],label='True '+channel,linewidth=2)
    axes[0].plot(x,path['fitted'][:,i],linestyle='--',label='Fitted '+channel)
    if path['lower95'] is not None:
        axes[0].fill_between(x,path['lower95'][:,i],path['upper95'][:,i],
                             alpha=.23,label='95% pointwise posterior interval')
    axes[1].plot(x,path['raw_spend'][:,int(channel[1:])-1],label='Observed raw input '+channel)
    axes[2].plot(x,path['truth'][:,0],label='True intrinsic baseline')
    axes[2].plot(x,path['fitted'][:,0],linestyle='--',label='Fitted intrinsic baseline')
    for axis in axes:
        axis.grid(alpha=.2); axis.legend(loc='best')
    axes[2].set_xlabel('Reported week (scored 7–103)')
    fig.suptitle(LABELS[regime] + f' · world {world}, {channel} · point '+('pass' if expected else 'fail'))
    display(fig); plt.close(fig)
"""),
            md("""## Scientific limits and original interpretations

See the **byte-identical historically approved** [`TV narrative`](tv50.md) and [`bounded narrative`](bounded50.md) for detailed interpretation and original exact report identities. This export is a retrospective compact numerical snapshot and reproducible analysis/presentation, **not historical model/sampler regeneration or a newly approved report**. It cannot retroactively certify original source/output bytes, posterior calibration, held-out or intervention performance, real-spend realism, structural non-identifiability or an algorithm ranking. The current PR1 pipeline generates *new* supported configurations and does **not** exactly replay the historical paired-TV intervention, bounded optimizer seed-per-world rule or original TV reconstruction RNG; PR1's short smoke fits are engineering checks, not data points here."""),
        ],
        metadata={
            "kernelspec": {
                "name": "recovery-pipeline",
                "display_name": "Recovery isolated Python",
                "language": "python",
            },
            "language_info": {"name": "python"},
        },
    )


def build(directory: Path, output: Path, kernel: str) -> None:
    book = NotebookClient(
        notebook(),
        timeout=120,
        kernel_name=kernel,
        resources={"metadata": {"path": str(directory)}},
    ).execute()
    # Never leave a partially executed report after an execution failure.
    nbf.write(book, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="recovery-pipeline")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    folder = Path(__file__).parent.resolve()
    build(folder, args.out.resolve() if args.out else folder / "report.ipynb", args.kernel)
