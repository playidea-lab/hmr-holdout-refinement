#!/usr/bin/env python3
"""exp790: 데이터 효율 곡선을 배우 분리 조건에서 다시 잰다.

논문 §IV의 "5시퀀스 = ceiling의 83~89%"는 시퀀스 LOO 분모를 쓴다. F1b가 그
분모를 3분의 1 부풀려진 것으로 보였으므로, 새 사람에 대한 배포에는 맞지 않는다.

여기서는 폴드를 배우로 잡는다:
  - 평가: held-out 배우의 전 프레임
  - 후보: 24개 시퀀스에서 held-out 배우의 행을 뺀 것. 남는 행이 없으면 제외
    (배우 1을 빼면 그 사람만 나오는 8개 시퀀스가 사라져 16개가 남는다)
  - 선택: 논문과 같이 시퀀스 평균 예측 포즈의 L2 거리로 가까운 k개
  - 분모: 같은 배우 분리 조건에서 후보 전부를 쓴 값 (= 배우 LOO ceiling)
"""

import os
import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parent.parent))
ANN = os.environ.get("TOKENHMR_3DPW_NPZ", "data/tokenhmr/3dpw_test.npz")
PATHS = {
    "TokenHMR": REPO / "data/tokenhmr_3dpw_cuda.pkl",
    "4DHumans": REPO / "data/4dhumans_3dpw_cuda.pkl",
}
K_VALUES = [1, 2, 3, 5, 10]
EPOCHS = 10


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        L, p = [], 42
        for h in (128, 128, 64):
            L += [nn.Linear(p, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(0.1)]
            p = h
        L.append(nn.Linear(p, 42))
        self.net = nn.Sequential(*L)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class DS(Dataset):
    def __init__(self, pr, gt):
        self.p = torch.from_numpy(pr.reshape(-1, 42)).float()
        self.g = torch.from_numpy(gt.reshape(-1, 42)).float()
        self.r = self.g - self.p

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return {"p": self.p[i], "g": self.g[i], "r": self.r[i]}


def sim(S1, S2):
    B = S1.shape[0]
    S1, S2 = S1.permute(0, 2, 1), S2.permute(0, 2, 1)
    m1, m2 = S1.mean(2, keepdim=True), S2.mean(2, keepdim=True)
    X1, X2 = S1 - m1, S2 - m2
    v1 = (X1 ** 2).sum(dim=(1, 2))
    K = torch.bmm(X1, X2.permute(0, 2, 1))
    U, _, Vh = torch.linalg.svd(K)
    Z = torch.eye(3, device=S1.device).unsqueeze(0).repeat(B, 1, 1)
    Z[:, -1, -1] = torch.sign(torch.linalg.det(torch.bmm(U, Vh)))
    R = torch.bmm(torch.bmm(Vh.permute(0, 2, 1), Z), U.permute(0, 2, 1))
    t = torch.bmm(R, K).diagonal(dim1=-2, dim2=-1).sum(-1)
    sc = (t / v1).unsqueeze(-1).unsqueeze(-1)
    return (sc * torch.bmm(R, S1) + (m2 - sc * torch.bmm(R, m1))).permute(0, 2, 1)


def pa(p, g):
    return torch.sqrt(((sim(p, g) - g) ** 2).sum(-1)).mean(-1) * 1000


def fit_eval(pred, gt, tr, te, dev):
    torch.manual_seed(42)
    m = MLP().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.L1Loss()
    dl = DataLoader(DS(pred[tr], gt[tr]), batch_size=256, shuffle=True, num_workers=2)
    for _ in range(EPOCHS):
        m.train()
        for b in dl:
            loss = crit(m(b["p"].to(dev)), b["r"].to(dev))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sch.step()
    m.eval()
    before, after = [], []
    with torch.no_grad():
        for b in DataLoader(DS(pred[te], gt[te]), batch_size=256, num_workers=2):
            p, g = b["p"].to(dev), b["g"].to(dev)
            r = p + m(p)
            before += pa(p.view(-1, 14, 3), g.view(-1, 14, 3)).cpu().tolist()
            after += pa(r.view(-1, 14, 3), g.view(-1, 14, 3)).cpu().tolist()
    return np.array(before), np.array(after)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ann = np.load(ANN, allow_pickle=True)
    imgnames, betas = ann["imgname"], ann["betas"]
    lut, subj = {}, []
    for b in betas:
        subj.append(lut.setdefault(tuple(np.round(b, 3)), len(lut)))
    subj = np.array(subj)
    tmp = defaultdict(list)
    for i, nm in enumerate(imgnames):
        tmp[nm.split("/")[1]].append(i)
    seqs = sorted(tmp)
    subjects = sorted(set(subj.tolist()))

    results = {}
    for name, path in PATHS.items():
        with open(path, "rb") as f:
            s = pickle.load(f)
        pred = np.stack([x["pred_joints_3d_eval"] for x in s])
        gt = np.stack([x["gt_joints_3d"] for x in s])
        n = len(pred)
        flat = pred.reshape(n, -1)
        idx = {q: np.array([i for i in tmp[q] if i < n]) for q in seqs}

        agg = defaultdict(lambda: ([], []))
        for u in subjects:
            te = np.array([i for i in range(n) if subj[i] == u])
            if len(te) == 0:
                continue
            # 후보 = 각 시퀀스에서 held-out 배우 행을 뺀 것
            cand = {}
            for q in seqs:
                keep = idx[q][subj[idx[q]] != u]
                if len(keep) > 0:
                    cand[q] = keep
            names = sorted(cand)
            mu_te = flat[te].mean(0)
            d = np.array([np.linalg.norm(mu_te - flat[cand[q]].mean(0)) for q in names])
            order = np.argsort(d)

            for k in K_VALUES + ["all"]:
                kk = len(names) if k == "all" else min(k, len(names))
                tr = np.concatenate([cand[names[i]] for i in order[:kk]])
                b, a = fit_eval(pred, gt, tr, te, dev)
                agg[k][0].append(b)
                agg[k][1].append(a)
            print(f"  {name} subject {u}: 후보 {len(names)}개 시퀀스, 평가 {len(te)}프레임",
                  flush=True)

        res = {}
        base = float(np.concatenate(agg["all"][0]).mean())
        ceil_after = float(np.concatenate(agg["all"][1]).mean())
        gap = base - ceil_after
        print(f"\n===== {name}  baseline {base:.3f}  배우LOO ceiling {ceil_after:.3f} "
              f"({100 * gap / base:+.2f}%)", flush=True)
        for k in K_VALUES + ["all"]:
            a = float(np.concatenate(agg[k][1]).mean())
            pct_gain = 100 * (base - a) / base
            pct_ceil = 100 * (base - a) / gap if gap > 0 else float("nan")
            res[str(k)] = {"after": a, "pct_gain": pct_gain, "pct_of_ceiling": pct_ceil}
            tag = f"k={k}" if k != "all" else "k=all"
            print(f"  {tag:6s} {a:7.3f}  개선 {pct_gain:+6.2f}%  "
                  f"ceiling의 {pct_ceil:5.1f}%", flush=True)
            if k in (1, 5):
                print(f"    @{name}_k{k}_pct_of_ceiling={pct_ceil:.2f}", flush=True)
        res["baseline"] = base
        res["ceiling_after"] = ceil_after
        results[name] = res

    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}/results.json", flush=True)


if __name__ == "__main__":
    main()
