"""Check the PR file scope and record checksums without committing or pushing."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
MISSION=ROOT/'mission5_ylj'


def main():
    paths=set()
    for flag in ['--cached','--others']:
        output=subprocess.check_output(['git','ls-files',flag,'--exclude-standard','-z','--','mission5_ylj'],cwd=ROOT)
        paths.update(p.decode('utf-8') for p in output.split(b'\0') if p)
    paths.discard('mission5_ylj/submission_manifest.json')
    entries=[]
    sensitive=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{24,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)')
    for relative in sorted(paths):
        path=ROOT/relative
        if not path.is_file():
            continue
        assert path.resolve().is_relative_to(MISSION.resolve()),relative
        assert path.suffix.lower() not in ['.pth','.npz'],relative
        assert not any(x in path.relative_to(MISSION).parts for x in ['.build','.idea','__pycache__']),relative
        assert path.stat().st_size<100*1024*1024,relative
        data=path.read_bytes()
        if path.suffix.lower() in ['.py','.md','.json','.txt','.mjs','.csv','.gitignore']:
            assert not sensitive.search(data),f'Potential credential pattern in {relative}'
        entries.append({'path':relative,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    experiments=list((MISSION/'experiments').glob('mission5_joint_*'))
    assert len(experiments)==6
    for exp in experiments:
        summary=json.loads((exp/'summary.json').read_text(encoding='utf-8'))
        assert summary['num_runs']==5
        assert all((exp/f'seed_{seed}/metrics.json').is_file() for seed in range(42,47))
    for name in ['mission5_report.pptx']:
        assert (MISSION/'ppt'/name).is_file(),name
    manifest={'scope':'mission5_ylj only','files':entries,'file_count':len(entries),
              'total_bytes':sum(e['bytes'] for e in entries),'self_excluded':True,
              'experiments':6,'seeds_per_experiment':5,
              'peer_sources':'comparison_inputs/download_manifest.json',
              'presentation':'ppt/mission5_report.pptx'}
    (MISSION/'submission_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Checked {len(entries)} files, {manifest["total_bytes"]/1024**2:.2f} MiB; 6 experiments / 30 seed results.')
    print('No dataset, model weight, downloaded peer folder, environment, IDE folder or build directory in PR scope.')


if __name__=='__main__':main()
