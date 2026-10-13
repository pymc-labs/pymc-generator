"""Static notebook template: ordinary saved tables and PNG plots, never widgets."""

from __future__ import annotations

import os
from pathlib import Path

import nbformat as nbf


def notebook(out: Path) -> dict:
    project = Path(__file__).resolve().parents[2]
    relative = os.path.relpath(project, out)
    cells = [
        nbf.v4.new_markdown_cell(
            "# Native recovery · sealed saved artifacts\n\n"
            "**Run All is read-only and never generates or fits.** The full configuration, methods, "
            "score denominators and inputs are sealed to this report ID. Values are descriptive; "
            "short fits are NOT convergence, identification or scientific approval."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport sys\n"
            f"sys.path.insert(0, str((Path.cwd() / {relative!r}).resolve()))\n"
            "from studies.recovery.reporting import read_report\n"
            "from IPython.display import display, Markdown\n"
            "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\n"
            "evidence = read_report(Path.cwd())\n"
            "display(Markdown('**Sealed run ID:** `' + evidence['run_id'] + '` · ' + str(evidence['provenance'])))"
        ),
        nbf.v4.new_markdown_cell("## Resolved configuration and complete per-method coverage"),
        nbf.v4.new_code_cell(
            "display(pd.json_normalize(evidence['resolved_config']).T.rename(columns={0:'resolved value'}))\n"
            "coverage = pd.DataFrame([{'method': method, 'worlds': len({r['world_id'] for r in evidence['worlds'] if r['method']==method}), "
            "'channel_rows': sum(c['method']==method for c in evidence['channels'])} "
            "for method in evidence['pattern_rates']])\ndisplay(coverage)"
        ),
        nbf.v4.new_markdown_cell(
            "## True oracle profiles · observation noise and scored-regime variation\n\n"
            "Baseline and contribution temporal moments use the exact saved scored truth; "
            "noise and parameter/reference values come from original generation inputs. "
            "These are not fit initialization or scientific explanation."
        ),
        nbf.v4.new_code_cell(
            "profiles = pd.json_normalize([{'world':r['world_id'],'method':r['method'],**r['profile']} "
            "for r in evidence['worlds']])\ndisplay(profiles)"
        ),
        nbf.v4.new_markdown_cell(
            "## Diagnostics, point gates and uncertainty are separate\n\n"
            "MCMC requires stringent R-hat/ESS/BFMI/divergence/depth screening; "
            "ADVI finite loss is NOT convergence (no meaningful R-hat/ESS), and its draw spread "
            "is not calibrated. The bounded OLS optimizer has no posterior intervals. "
            "Pointwise posterior intervals do not certify out-of-sample calibration."
        ),
        nbf.v4.new_code_cell(
            "diagnostics = pd.json_normalize([{'world':r['world_id'],'method':r['method'],"
            "'status':r['computational_status'],**r['diagnostics']} for r in evidence['worlds']])\n"
            "display(diagnostics)\n"
            "recovery = pd.DataFrame([{'world':r['world_id'],'method':r['method'],"
            "'status':r['computational_status'],'media':r['recovery']['media_pass'],"
            "'controls':r['recovery']['control_pass'],'joint':r['recovery']['joint_pass'],"
            "'total_media':(r['recovery']['total_media'] or {}).get('pass'),"
            "'baseline':(r['recovery']['baseline'] or {}).get('pass'),"
            "'uncertainty':r['uncertainty']['type']} for r in evidence['worlds']])\n"
            "display(recovery)"
        ),
        nbf.v4.new_markdown_cell(
            "## Pattern-bearing channel failures and denominators\n\n"
            "Overlapping labels and upward/downward trends are distinct. Known-zero effects "
            "remain in `total`, not `eligible`. Invalid fits are unresolved, not observed failures; "
            "zero denominators are **not represented**, never a zero failure rate."
        ),
        nbf.v4.new_code_cell(
            "for method, rates in evidence['pattern_rates'].items():\n"
            "    display(Markdown('### ' + method))\n"
            "    display(pd.DataFrame(rates)[['kind','pattern','total','eligible','known_zero',"
            "'evaluable','failed','unresolved','failure_rate','status','small_count_warning']])\n"
            "display(pd.DataFrame(evidence['channels']))"
        ),
        nbf.v4.new_markdown_cell(
            "## Static true/fitted examples with raw input and baseline\n\n"
            "**Explicit selection policy:** display every saved method for the first TWO world IDs "
            "(or fewer if unavailable); in each, select the FIRST nonzero media channel (C1, C2, …). "
            "All other worlds/channels are in `evidence.json` and the complete tables above. "
            "These normal PNG outputs require no live kernel, browser callbacks or JavaScript. "
            "The shaded 95% interval is pointwise and shown only where saved. "
            "If selected draws are nonfinite, fitted paths/intervals are UNAVAILABLE; "
            "the static plot shows true paths and raw input without inventing fitted values."
        ),
        nbf.v4.new_code_cell(
            "selected_worlds = sorted({row['world_id'] for row in evidence['worlds']})[:2]\n"
            "for key, t in sorted(evidence['trajectories'].items(), key=lambda pair:(int(pair[0].split(':')[0]),pair[0])):\n"
            "    if int(key.split(':')[0]) not in selected_worlds: continue\n"
            "    truth = np.asarray(t['truth'])\n"
            "    fit = np.asarray(t['fitted']) if t['fitted'] is not None else None\n"
            "    options = [(i, name) for i,name in enumerate(t['terms']) if name.startswith('C') and np.linalg.norm(truth[:,i])>0]\n"
            "    if not options: raise ValueError('missing required nonzero media channel')\n"
            "    ix, name = options[0]\n"
            "    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, layout='constrained')\n"
            "    weeks = t['weeks']\n"
            "    axes[0].plot(weeks,truth[:,ix],label='True '+name)\n"
            "    if fit is not None: axes[0].plot(weeks,fit[:,ix],'--',label='Fitted '+name)\n"
            "    else: axes[0].text(.02,.95,'Fitted contribution unavailable (nonfinite selected draws)',transform=axes[0].transAxes,va='top')\n"
            "    if t['lower95'] is not None:\n"
            "        axes[0].fill_between(weeks,np.asarray(t['lower95'])[:,ix],np.asarray(t['upper95'])[:,ix],alpha=.2,label='95% pointwise interval')\n"
            "    axes[1].plot(weeks,np.asarray(t['raw_spend'])[:,int(name[1:])-1],label='Raw input '+name)\n"
            "    axes[2].plot(weeks,truth[:,0],label='True baseline')\n"
            "    if fit is not None: axes[2].plot(weeks,fit[:,0],'--',label='Fitted baseline')\n"
            "    for axis in axes: axis.legend(loc='best'); axis.grid(alpha=.2)\n"
            "    axes[2].set_xlabel('Week')\n"
            "    fig.suptitle(key+' · '+name+' · '+next(r['computational_status'] for r in evidence['worlds'] if str(r['world_id'])+':'+r['method']==key))\n"
            "    display(fig)\n    plt.close(fig)"
        ),
        nbf.v4.new_markdown_cell(
            "## Run-bound human interpretation · PENDING\n\nThe sealed interpretation template is not an approved scientific conclusion."
        ),
        nbf.v4.new_code_cell("display(Markdown(Path('interpretation.md').read_text()))"),
    ]
    return dict(
        nbf.v4.new_notebook(
            cells=cells,
            metadata={
                "kernelspec": {
                    "display_name": "Recovery isolated Python",
                    "language": "python",
                    "name": "recovery-pipeline",
                },
                "language_info": {"name": "python"},
            },
        )
    )
