"""분리 split 검증: 3DPW train(+val) 을 라벨 풀로 쓰고 test 에서만 평가한다 (HMR 2.0).

정제기·손실·옵티마이저·스케줄은 exp775/exp803b 와 같다(RefineMLP, L1, AdamW 1e-3, cosine, 10 epoch).
입력은 infer_hmr2.py 가 같은 npz 행 순서로 저장한 14관절 예측·GT 다.
(a) 풀 전체 학습 → test, (b) 고정 무작위 k 시퀀스(20회 추첨), (c) test 시퀀스별 최근접 k (transductive).
배우: sequenceFile betas(소수 3자리) 로 신원을 만들고 split 간 겹침을 확인한다.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import exp803b_data_efficiency_stats as H  # noqa: E402

D = Path(os.path.join(os.environ.get("RESULTS_DIR", "outputs"), "clean_split"))
DEV = torch.device("cuda")
BATCH, LR, WD, EPOCHS = 256, 1e-3, 1e-4, 10
N_SEEDS, N_DRAWS = 5, 20
KS_FIXED, KS_NEAR = (1, 2, 5, 10), (1, 5)


def pa_torch(pred, gt):
    """exp775 와 같은 표준 Procrustes PA-MPJPE(mm)."""
    p = pred.reshape(-1, 14, 3).double()
    g = gt.reshape(-1, 14, 3).double()
    pc, gc = p - p.mean(1, keepdim=True), g - g.mean(1, keepdim=True)
    sp = pc.pow(2).sum((1, 2), keepdim=True).sqrt()
    sg = gc.pow(2).sum((1, 2), keepdim=True).sqrt()
    pc, gc = pc / (sp + 1e-8), gc / (sg + 1e-8)
    U, S, Vh = torch.linalg.svd(pc.transpose(1, 2) @ gc)
    V = Vh.transpose(1, 2)
    d = torch.sign(torch.linalg.det(V @ U.transpose(1, 2)))
    Dm = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], 1))
    R = V @ Dm @ U.transpose(1, 2)
    scale = (S[:, 0] + S[:, 1] + d * S[:, 2]).view(-1, 1, 1)
    al = sg * scale * (pc @ R.transpose(1, 2)) + g.mean(1, keepdim=True)
    return ((al - g).pow(2).sum(-1).sqrt().mean(-1) * 1000.0).float()


def train(X, Y, seed):
    torch.manual_seed(seed)
    model = H.RefineMLP().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = len(X)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            nn.functional.l1_loss(model(X[b]), Y[b]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    model.eval()
    return model


@torch.no_grad()
def errs(model, X, Y):
    return pa_torch(model(X), Y).cpu().numpy()


def load(split):
    a = np.load(D / f"3dpw_{split}_seqfmt.npz", allow_pickle=True)
    p = np.load(D / f"hmr2_{split}.npz")
    assert len(a["imgname"]) == len(p["pred"])
    X = torch.tensor(p["pred"].reshape(len(p["pred"]), -1), dtype=torch.float32, device=DEV)
    Y = torch.tensor(p["gt"].reshape(len(p["gt"]), -1), dtype=torch.float32, device=DEV)
    ids = [tuple(np.round(b, 3)) for b in a["betas"]]
    return dict(X=X, Y=Y, seq=a["seq"], ids=ids, pred=p["pred"])


def main():
    te, tr, va = load("test"), load("train"), load("validation")
    base = errs(lambda x: x, te["X"], te["Y"])
    print(f"test 기준 PA-MPJPE {base.mean():.3f} mm (논문 54.33) @baseline={base.mean():.4f}", flush=True)

    # 배우·장면 분리 확인
    sid = {s: set() for s in ("train", "validation", "test")}
    for nm, d in (("train", tr), ("validation", va), ("test", te)):
        sid[nm] = set(d["ids"])
    print(f"배우 수 train {len(sid['train'])} val {len(sid['validation'])} test {len(sid['test'])}; "
          f"train∩test {len(sid['train'] & sid['test'])}, val∩test {len(sid['validation'] & sid['test'])}, "
          f"train∩val {len(sid['train'] & sid['validation'])}", flush=True)
    sc = lambda d: {s.rsplit('_', 1)[0] for s in d["seq"]}  # noqa: E731
    print(f"장면(시퀀스 접두) 겹침 train∩test {sorted(sc(tr) & sc(te))}, val∩test {sorted(sc(va) & sc(te))}", flush=True)

    # 풀 = train+val
    PX = torch.cat([tr["X"], va["X"]])
    PY = torch.cat([tr["Y"], va["Y"]])
    pseq = np.concatenate([tr["seq"], va["seq"]])
    pool = sorted(set(pseq))
    rows = {s: np.nonzero(pseq == s)[0] for s in pool}
    print(f"풀 시퀀스 {len(pool)}개, {len(PX)}프레임", flush=True)
    rel = lambda e: 100.0 * (base.mean() - e.mean()) / base.mean()  # noqa: E731

    # (a) 풀 전체, train 만
    for nm, (x, y) in {"train+val": (PX, PY), "train": (tr["X"], tr["Y"])}.items():
        r = [rel(errs(train(x, y, s), te["X"], te["Y"])) for s in range(N_SEEDS)]
        print(f"(a) {nm} 전체 → test: {np.mean(r):.2f} ± {np.std(r):.2f}% @a_{nm.replace('+', '_')}={np.mean(r):.3f}", flush=True)

    # (b) 고정 무작위 k 시퀀스 — 추첨 j 는 시드 j
    rng = np.random.default_rng(0)
    for k in KS_FIXED:
        r = []
        for j in range(N_DRAWS):
            pick = rng.choice(len(pool), k, replace=False)
            idx = torch.as_tensor(np.concatenate([rows[pool[i]] for i in pick]), device=DEV)
            r.append(rel(errs(train(PX[idx], PY[idx], j), te["X"], te["Y"])))
        r = np.array(r)
        print(f"(b) 고정 무작위 k={k}: {r.mean():.2f} ± {r.std():.2f}% (범위 {r.min():.2f}–{r.max():.2f}, 손해 {np.mean(r < 0) * 100:.0f}%) "
              f"@b_k{k}={r.mean():.3f}", flush=True)

    # (c) test 시퀀스별 최근접 k — 예측 평균 거리
    pmean = {s: np.concatenate([tr["pred"], va["pred"]])[rows[s]].reshape(len(rows[s]), -1).mean(0) for s in pool}
    tseq = sorted(set(te["seq"]))
    for k in KS_NEAR:
        res = []
        for seed in range(N_SEEDS):
            after = np.zeros(len(base), np.float32)
            for t in tseq:
                ti = np.nonzero(te["seq"] == t)[0]
                tm = te["pred"][ti].reshape(len(ti), -1).mean(0)
                near = sorted(pool, key=lambda s: np.linalg.norm(pmean[s] - tm))[:k]
                idx = torch.as_tensor(np.concatenate([rows[s] for s in near]), device=DEV)
                m = train(PX[idx], PY[idx], seed)
                tit = torch.as_tensor(ti, device=DEV)
                after[ti] = errs(m, te["X"][tit], te["Y"][tit])
            res.append(rel(after))
        print(f"(c) 최근접 k={k} (transductive): {np.mean(res):.2f} ± {np.std(res):.2f}% @c_k{k}={np.mean(res):.3f}", flush=True)
    print("@done=1", flush=True)


if __name__ == "__main__":
    main()
