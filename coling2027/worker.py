"""Complete one predeclared seed alongside the sequential suite.

No experimental settings change. Different seeds write separate directories.
The sequential runner skips these completed results when it reaches seed 44.
Concurrent wall times must not be presented as isolated GPU benchmarks.
"""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--seed',type=int,default=44,choices=[43,44])
args=parser.parse_args()
plan=json.loads((root/'suite_plan.json').read_text())
for i,job in enumerate(plan['jobs']):
    if job['seed']!=args.seed:
        continue
    cmd=[sys.executable,'-u',str(root/'run.py'),'--revision',plan['model_revision']]
    for k,v in job.items():
        cmd.extend(['--'+k,str(v)])
    label=f"worker_{i:02d}_{job['dataset']}_{job['n']}_{job['rate']}_{job['mode']}"
    print('START',label,flush=True)
    start=time.monotonic()
    with (root/'logs'/f'{label}.log').open('a') as f:
        r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=root.parent,
                         env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'})
    record={'job':job,'exit_code':r.returncode,'seconds':time.monotonic()-start,'concurrent_seed_worker':True}
    with (root/'worker_status.jsonl').open('a') as f:
        f.write(json.dumps(record)+'\n')
    print('END',label,r.returncode,round(record['seconds'],1),flush=True)
    if r.returncode:
        print((root/'logs'/f'{label}.log').read_text()[-4000:],flush=True)
        sys.exit(r.returncode)
