"""Read-only aggregation of recorded evidence; never imports a generator/model."""
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
EVIDENCE=ROOT/'study-200k-evidence'
ledger=json.loads((EVIDENCE/'ledger.json').read_text())
rows=[]
for entry in ledger['attempts']:
    folder=EVIDENCE/entry['name']; result=folder/'result.json'
    if not result.exists():
        rows.append(dict(name=entry['name'],envelope_status=entry['status'],exit_code=entry.get('exit_code'),result='incomplete'))
        continue
    r=json.loads(result.read_text())
    args=json.loads((folder/'environment.json').read_text())['args'] if (folder/'environment.json').exists() else {}
    row=dict(name=entry['name'],approach=entry['approach'],phase=r.get('phase',args.get('phase')),
        mode=r.get('mode',args.get('mode')),seed_contract=r.get('seed_contract',args.get('seed_contract','legacy')),
        envelope_status=entry['status'],exit_code=entry.get('exit_code'),result_status=r.get('status'),
        elapsed_s=entry.get('elapsed_s'),source_directory=str(folder/'sources'))
    if r.get('phase') in ('primary','pilot'):
        n=r['candidates']; a=r['accepted']; p=a/n; z=1.96
        denominator=1+z*z/n
        center=(p+z*z/(2*n))/denominator
        half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
        cohorts=r.get('cohorts',[])
        bytes_=sum(c.get('bytes',0) for c in r['chunks'])
        setup=r['stage_times']['build_s']+r['stage_times']['compile_s']
        verified=r.get('verified_service_s',r['accepted_persisted_service_s']+r['readback_and_check_s'])
        steady=verified-setup
        cleanup=r.get('post_chunk_cleanup_and_progress_s')
        row.update(candidates=n,accepted=a,rejected=n-a,acceptance_wilson95=[center-half,center+half],
            primary_elapsed_s=r.get('elapsed_s'),cleanup_and_progress_s=cleanup,
            steady_including_cleanup_accepted_per_s=a/(steady+cleanup) if cleanup is not None else None,
            projected_200k_local_including_cleanup_s=setup+(steady+cleanup)*200000/a if cleanup is not None and a else None,
            setup_s=setup,model_release_s=r['stage_times'].get('model_release_s'),
            verified_service_s=verified,steady_verified_s=steady,steady_verified_accepted_per_s=a/steady,
            cohorts=cohorts,primary_memory=r['primary_memory'],corpus_bytes=bytes_,bytes_per_accepted=bytes_/a if a else None,
            candidate_family_counts=r['candidate_family_counts'],accepted_family_counts=r['accepted_family_counts'],
            input_counts=r['input_counts'],component_counts=r['component_counts'],
            accepted_input_counts=r['accepted_input_counts'],accepted_component_counts=r['accepted_component_counts'],
            rejection_reasons=r.get('rejection_reasons'),compile_records=r['compile_records'],rng_work=r.get('rng_work'),
            rebuild_pass=(folder/'rebuild.json').exists() and json.loads((folder/'rebuild.json').read_text())['status']=='pass',
            projected_candidates_200k=200000*n/a if a else None,
            # Conditional local forecast only: once-per-bank setup plus this
            # measured steady pipeline; excludes disk/upload/parallel effects.
            projected_200k_local_verified_s=setup+steady*200000/a if a else None,
            projected_200k_corpus_bytes=bytes_*200000/a if a else None)
    elif r.get('phase')=='oracle':
        row.update(families=[x['family'] for x in r['families']],oracle_memory=r['memory'])
    elif r.get('phase')=='seed_audit': row['seed_audit']=r
    rows.append(row)
summary=dict(generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),ledger=ledger,attempts=rows,
    caveats=['Forecasts are conditional local tmpfs compression/write/readback, not storage/network/Modal estimates.',
        'Model-release/turnover costs can enter first-per-family cohort steady timings; inspect both cohorts and stage records.',
        'Legacy and logical-ID cohorts are distinct seed policies; no sole execution-mode attribution across policies.',
        'Missing rebuild marker or failed envelope is retained explicitly, not promoted to complete correctness.'])
(ROOT/'study-200k-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
manifest=[]
for path in sorted(EVIDENCE.rglob('*')):
    if path.is_file() and path.name not in ('.lock','artifact-manifest.json'):
        manifest.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(EVIDENCE/'artifact-manifest.json').write_text(json.dumps(dict(files=manifest,total_bytes=sum(f['bytes'] for f in manifest)),indent=2)+'\n')
print(json.dumps([dict(name=r['name'],candidates=r.get('candidates'),accepted=r.get('accepted'),
    steady_verified_per_s=r.get('steady_verified_accepted_per_s'),rebuild=r.get('rebuild_pass'),
    envelope=r['envelope_status']) for r in rows],indent=2))
