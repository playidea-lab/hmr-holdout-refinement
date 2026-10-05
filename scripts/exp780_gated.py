#!/usr/bin/env python3
"""exp780: 보정을 걸지 말지 고르는 게이트를 실제로 구현한다.

논문은 저오차 표본의 약 3분의 1이 보정으로 악화된다는 것을 보고하고 게이트를
권고하지만 시험하지 않는다. 여기서 시험한다.

게이트 신호는 refiner 자신의 보정 크기 ||g(J)||다. 정답이 필요 없고 추가 모델도
없다. refiner가 크게 고치려 든다는 것은 그 예측이 자기가 배운 오차 패턴에서
멀다는 뜻이고, 작게 고치려 든다는 것은 이미 맞다고 보는 것이다. 후자를 건드리지
않는다.

임계값은 학습 폴드에서만 정한다(그 폴드의 보정 크기 분위수). held-out에는
그 값을 그대로 적용하므로 평가 누출이 없다.
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
    "SPIN": REPO / "data/spin_3dpw_14j.pkl",
}
QUANTILES = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
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


@torch.no_grad()
def infer(model, pr, gt, dev, bs=256):
    """반환: before, after(전량 보정), 보정 크기."""
    b, a, d = [], [], []
    for k in range(0, len(pr), bs):
        p = torch.from_numpy(pr[k:k + bs].reshape(-1, 42)).float().to(dev)
        g = torch.from_numpy(gt[k:k + bs].reshape(-1, 42)).float().to(dev)
        delta = model(p)
        r = p + delta
        b += pa(p.view(-1, 14, 3), g.view(-1, 14, 3)).cpu().tolist()
        a += pa(r.view(-1, 14, 3), g.view(-1, 14, 3)).cpu().tolist()
        d += delta.norm(dim=-1).cpu().tolist()
    return np.array(b), np.array(a), np.array(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    imgnames = np.load(ANN, allow_pickle=True)["imgname"]
    tmp = defaultdict(list)
    for i, nm in enumerate(imgnames):
        tmp[nm.split("/")[1]].append(i)

    results = {}
    for name, path in PATHS.items():
        with open(path, "rb") as f:
            s = pickle.load(f)
        pred = np.stack([x["pred_joints_3d_eval"] for x in s])
        gt = np.stack([x["gt_joints_3d"] for x in s])
        n = len(pred)
        idx = {k: np.array([i for i in v if i < n]) for k, v in tmp.items()}
        seqs = sorted(idx)

        B = np.zeros(n); A = np.zeros(n); THR = np.zeros((n, len(QUANTILES)))
        for sq in seqs:
            te = idx[sq]
            tr = np.concatenate([v for k, v in idx.items() if k != sq])
            torch.manual_seed(42)
            m = MLP().to(dev)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
            crit = nn.L1Loss()
            dl = DataLoader(DS(pred[tr], gt[tr]), batch_size=256, shuffle=True,
                            num_workers=2)
            for _ in range(EPOCHS):
                m.train()
                for bb in dl:
                    loss = crit(m(bb["p"].to(dev)), bb["r"].to(dev))
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
                sch.step()
            m.eval()
            # 임계값은 학습 폴드의 보정 크기 분위수로만 정한다
            _, _, d_tr = infer(m, pred[tr], gt[tr], dev)
            thr = np.quantile(d_tr, QUANTILES)
            b, a, d_te = infer(m, pred[te], gt[te], dev)
            B[te] = b; A[te] = a
            for j, t in enumerate(thr):
                THR[te, j] = (d_te >= t)   # 게이트 통과 = 보정 적용

        # 사분위(초기 PA-MPJPE 기준) — Q1 악화율을 보기 위해
        q = np.quantile(B, [0.25, 0.5, 0.75])
        bins = np.digitize(B, q)
        res = {"baseline": float(B.mean())}
        print(f"===== {name}  baseline {B.mean():.3f}", flush=True)
        for j, ql in enumerate(QUANTILES):
            keep = THR[:, j].astype(bool)
            out_pa = np.where(keep, A, B)
            worse = float((out_pa > B).mean() * 100)
            q1worse = float((out_pa[bins == 0] > B[bins == 0]).mean() * 100)
            pct = 100 * (B.mean() - out_pa.mean()) / B.mean()
            res[f"gate_q{int(ql * 100)}"] = {
                "refined_frac": float(keep.mean() * 100), "pa": float(out_pa.mean()),
                "pct": pct, "worse_pct": worse, "q1_worse_pct": q1worse}
            tag = "게이트 없음" if ql == 0.0 else f"하위 {int(ql * 100)}% 보류"
            print(f"  {tag:<16s} 보정적용 {keep.mean() * 100:5.1f}%  "
                  f"PA {out_pa.mean():6.3f} ({pct:+5.2f}%)  "
                  f"악화 {worse:5.1f}%  Q1악화 {q1worse:5.1f}%", flush=True)
        results[name] = res
        g0, gb = res["gate_q0"], res[f"gate_q30"]
        print(f"  @{name}_ungated_pa={g0['pa']:.4f} "
              f"@{name}_ungated_worse_pct={g0['worse_pct']:.2f} "
              f"@{name}_gated_pa={gb['pa']:.4f} "
              f"@{name}_gated_worse_pct={gb['worse_pct']:.2f}", flush=True)

    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}/results.json", flush=True)


if __name__ == "__main__":
    main()
