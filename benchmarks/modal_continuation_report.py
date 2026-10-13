"""Summarize committed-schema evidence without rerunning scientific computation."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT/'modal-continuation-evidence'
OLD = ROOT/'modal-targeted-evidence/attempt3'
rows = []
for arm in json.loads((OUT/'arms.json').read_text()):
    path = OUT/arm['arm']
    assert arm['status']=='pass'
    assert json.loads((path/'result.json').read_text())['status']=='pass'
    env = json.loads((path/'environment.json').read_text())
    assert env['affinity']==[0]
    assert env['cgroup_controls']['memory.max']==str(2*1024**3)
    assert env['cgroup_controls']['memory.swap.max']=='0'
    assert env['cgroup_controls']['cpu.max']=='100000 100000'
    for label in ['first','group10','group100']:
        row = json.loads((path/f'{label}.json').read_text())
        assert row['replay_pass']
        assert row['candidates']==2*row['draws']
        assert row['accepted']+row['rejected']==row['candidates']
        rows.append(dict(arm=arm['arm'],**row))
comparisons = []
for method in ['ordinary','dynamic','specialized']:
    for label in ['first','group10','group100']:
        a = OUT/f'3-{method}-r0'/f'{label}.npz'
        b = OUT/f'3-{method}-r1'/f'{label}.npz'
        ha,hb = [hashlib.sha256(p.read_bytes()).hexdigest() for p in [a,b]]
        assert ha==hb
        comparisons.append(dict(left=str(a.relative_to(ROOT)),right=str(b.relative_to(ROOT)),sha256=ha))
# Match all overlapping primary outputs from the original envelope as well.
for method,workload in [('dynamic',0),('ordinary',3)]:
    for label in ['first','group10','group100']:
        a = OLD/f'{workload}-{method}'/f'{label}.npz'
        b = OUT/f'{workload}-{method}-r0'/f'{label}.npz'
        if not a.exists():
            continue
        ha,hb = [hashlib.sha256(p.read_bytes()).hexdigest() for p in [a,b]]
        assert ha==hb
        comparisons.append(dict(left=str(a.relative_to(ROOT)),right=str(b.relative_to(ROOT)),sha256=ha))
summary = dict(rows=rows,archive_identity_checks=comparisons,
    candidate_evaluations=sum(r['candidates'] for r in rows),
    accepted_evaluations=sum(r['accepted'] for r in rows),
    rejected_evaluations=sum(r['rejected'] for r in rows),
    peak_self_rss_kib=max(r['memory']['self_peak_rss_kib'] for r in rows),
    peak_scope_bytes=max(r['memory']['scope_peak_bytes'] for r in rows))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines = ['| Workload / method | Draws per structure | Pipeline seconds (observed) | Build + compile seconds | Draw seconds |',
         '|---|---:|---:|---:|---:|']
for workload in [0,3]:
    for method in ['ordinary','dynamic','specialized']:
        for draws in [1,10,100]:
            group = [r for r in rows if r['workload']==workload and r['method']==method and r['draws']==draws]
            if not group:
                continue
            values = [[r['pipeline_s'],r['stages']['build_s']+r['stages']['compile_s'],r['stages']['draw_s']] for r in group]
            cells = []
            for i in range(3):
                v = [x[i] for x in values]
                cells.append(f'{min(v):.3f}–{max(v):.3f}' if len(v)>1 else f'{v[0]:.3f}')
            lines.append(f'| {"sparse" if workload==0 else "dense"} / {method} | {draws} | '+ ' | '.join(cells)+' |')
(ROOT/'modal-continuation-tables.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['rows','archive_identity_checks']},indent=2))
print('Archive identity checks:',len(comparisons))
print('\n'.join(lines))
