"""분리 split 의 배우 겹침: test 배우별로 풀(train+val)에 그 배우가 있는지, 있으면 몇 프레임인지,
그리고 (a) 풀 전체 학습의 배우별 개선율. 겹친 배우를 풀에서 빼면 어떻게 되는지도 잰다.
"""
import numpy as np
import torch

import clean_split as C

te, tr, va = C.load("test"), C.load("train"), C.load("validation")
base = C.errs(lambda x: x, te["X"], te["Y"])
PX = torch.cat([tr["X"], va["X"]])
PY = torch.cat([tr["Y"], va["Y"]])
pids = tr["ids"] + va["ids"]
tids = te["ids"]
uniq = sorted(set(tids), key=lambda u: -tids.count(u))
lab = {u: i for i, u in enumerate(uniq)}
pool_cnt = {u: sum(1 for p in pids if p == u) for u in uniq}
tarr = np.array([lab[u] for u in tids])
parr = np.array([lab.get(p, -1) for p in pids])

E_all = np.stack([C.errs(C.train(PX, PY, s), te["X"], te["Y"]) for s in range(C.N_SEEDS)])
for u in uniq:
    i = lab[u]
    m = tarr == i
    r = 100 * (base[m].mean() - E_all[:, m].mean(1)) / base[m].mean()
    # 이 배우를 풀에서 뺀 학습
    keep = torch.as_tensor(np.nonzero(parr != i)[0], device=C.DEV)
    r2 = [100 * (base[m].mean() - C.errs(C.train(PX[keep], PY[keep], s), te["X"][torch.as_tensor(np.nonzero(m)[0], device=C.DEV)],
                                          te["Y"][torch.as_tensor(np.nonzero(m)[0], device=C.DEV)]).mean()) / base[m].mean()
          for s in range(C.N_SEEDS)] if pool_cnt[u] else [np.nan]
    print(f"test 배우 {i}: test {m.sum():6d}프레임, 풀 내 {pool_cnt[u]:6d}프레임 | 풀 전체 {r.mean():.2f}±{r.std():.2f}% | "
          f"이 배우 제외 {np.nanmean(r2):.2f}±{np.nanstd(r2):.2f}%", flush=True)
seen = np.isin(tarr, [lab[u] for u in uniq if pool_cnt[u]])
for nm, m in (("풀에 있는 배우", seen), ("풀에 없는 배우", ~seen)):
    r = 100 * (base[m].mean() - E_all[:, m].mean(1)) / base[m].mean()
    print(f"{nm}: test {m.sum()}프레임, 개선 {r.mean():.2f}±{r.std():.2f}% @{'seen' if nm.startswith('풀에 있') else 'unseen'}={r.mean():.3f}", flush=True)
