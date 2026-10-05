"""exp775 를 (백본, 대조군) 단위로 나눠 한 GPU 에서 동시에 돌린다.

정제기는 34K 파라미터라 GPU 를 거의 못 채운다(메모리 31MB). 프로세스 여러 개를 함께 돌리면
벽시계 시간이 준다. 각 프로세스는 (a) 프로토콜을 다시 재므로(약 3.5분) 서로 독립이다.
CQ_PARAMS: jobs="TokenHMR:b,TokenHMR:c,..." n_seeds n_draws
"""

import json
import os
import subprocess
import sys
import time

prm = json.loads(os.environ.get("CQ_PARAMS") or "{}")
jobs = [j.split(":") for j in prm["jobs"].split(",")]
procs = []
t0 = time.time()
for bb, tag in jobs:
    p = {"n_seeds": prm.get("n_seeds", 5), "n_draws": prm.get("n_draws", 10), "backbones": bb,
         "parts": f"a,{tag}", "tag": f"{bb}_{tag}", "n_boot": 2000}
    env = {**os.environ, "CQ_PARAMS": json.dumps(p)}
    log = open(os.path.join(os.environ.get("RESULTS_DIR", "outputs"), f"exp775/log_{bb}_{tag}.txt"), "w")
    procs.append((bb, tag, subprocess.Popen([sys.executable, "-u", "scripts/exp775_share_seeds.py"],
                                            env=env, stdout=log, stderr=subprocess.STDOUT), log))
    print(f"시작 {bb}:{tag}", flush=True)
while procs:
    time.sleep(60)
    alive = []
    for bb, tag, pr, log in procs:
        if pr.poll() is None:
            alive.append((bb, tag, pr, log))
        else:
            log.close()
            print(f"종료 {bb}:{tag} rc={pr.returncode} ({(time.time()-t0)/60:.1f} min)", flush=True)
            with open(os.path.join(os.environ.get("RESULTS_DIR", "outputs"), f"exp775/log_{bb}_{tag}.txt")) as f:
                lines = f.read().splitlines()
            for ln in lines:
                if ln.startswith("  [몫]") or ln.startswith("  @") or "subject_" in ln and "[a]" in ln:
                    print(f"  {bb}:{tag} | {ln.strip()}", flush=True)
    if alive:
        tails = []
        for bb, tag, _, _ in alive:
            with open(os.path.join(os.environ.get("RESULTS_DIR", "outputs"), f"exp775/log_{bb}_{tag}.txt")) as f:
                ls = f.read().splitlines()
            tails.append(f"{bb}:{tag}→{ls[-1][:60] if ls else ''}")
        print(f"[{(time.time()-t0)/60:.0f} min] 진행: " + " | ".join(tails), flush=True)
    procs = alive
print(f"전부 종료 {(time.time()-t0)/60:.1f} min", flush=True)
