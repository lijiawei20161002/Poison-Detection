"""Predeclared suite. Persist plan before running; retain every failed run."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
seeds = [42,43,44]
jobs = []
for seed in seeds:
    for dataset,attack,n,rates in [
        ("sst2","scpn",6000,[0.,.01,.05]),
        ("imdb","cf",1000,[0.,.01,.05]),
        ("imdb","cf",10000,[0.,.005,.01,.05]),
    ]:
        for rate in rates:
            job = dict(dataset=dataset,attack=attack,n=n,rate=rate,seed=seed,mode="full")
            # Equal 5% removal budgets; same model, epoch count, optimizer, seed.
            if n >= 6000 and rate == .05:
                job["remove"] = "PD_KL,PD_observed,ZScore_released_unigram,random,oracle"
            jobs.append(job)
    for dataset,attack,n in [("sst2","scpn",6000),("imdb","cf",10000)]:
        for rate in [0.,.05]:
            jobs.append(dict(dataset=dataset,attack=attack,n=n,rate=rate,seed=seed,mode="lora"))
plan = {"created_utc":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        "epochs":10,"batch":64,"lr":.0003,"max_length":128,
        "model":"google/flan-t5-small","model_revision":"0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab",
        "jobs":jobs,"notes":"Plan fixed after N=1000 SCPN pilot; all seeds, failures, and negative results retained."}
path = ROOT / "suite_plan.json"
if path.exists():
    if json.loads(path.read_text())["jobs"] != jobs:
        raise RuntimeError("Refusing to overwrite a different registered suite")
else:
    path.write_text(json.dumps(plan,indent=2)+'\n')
(ROOT / "logs").mkdir(exist_ok=True)
for i,job in enumerate(jobs):
    cmd = [sys.executable,"-u",str(ROOT / "run.py"),"--revision",plan["model_revision"]]
    for key,value in job.items():
        cmd.extend(["--"+key,str(value)])
    label = f"{i:02d}_{job['dataset']}_{job['n']}_{job['rate']}_{job['seed']}_{job['mode']}"
    print(f"START {i+1}/{len(jobs)} {label}",flush=True)
    t = time.monotonic()
    with (ROOT / "logs" / f"{label}.log").open('a') as log:
        result = subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT.parent,
                                env={**os.environ,"OMP_NUM_THREADS":"4","OPENBLAS_NUM_THREADS":"4"})
    print(f"END {label} exit={result.returncode} seconds={time.monotonic()-t:.1f}",flush=True)
    with (ROOT / "suite_status.jsonl").open('a') as f:
        f.write(json.dumps({"job":job,"exit_code":result.returncode,"seconds":time.monotonic()-t})+'\n')
    if result.returncode:
        print((ROOT / "logs" / f"{label}.log").read_text()[-5000:],flush=True)
        sys.exit(result.returncode)
