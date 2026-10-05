#!/usr/bin/env python3
"""exp762: 논문 §III-D의 ablation 블록을 정정된 구현으로 다시 잰다.

왜: 기존 블록의 수치를 49.1x(1-x)로 역산하면 전부 38.4~38.7 대역이다.
    재측정된 기준값은 39.27mm(20.07%)이므로 그 블록은 폐기된 실행의 산출물이다.
    이 블록이 "34K에서 포화하므로 용량이 제약이 아니다"(F2)를 떠받치기 때문에
    ceiling 주장의 근거가 여기에 걸려 있다.

프로토콜은 exp761과 동일: TokenHMR, 3DPW 24시퀀스 LOO, AdamW(1e-3, 1e-4),
cosine, batch 256, grad clip 1.0, 출력층 zero-init. 각 설정을 시드 3개로 돌려
평균과 표준편차를 낸다 — 리뷰어가 "시드 편차보다 작은 차이로 결론을 냈다"고
지적한 항목들을 판정하려면 오차막대가 있어야 한다.

기준점(128-128-64, L1, 10 epochs, 23 시퀀스)은 exp761의 20.07±0.19%와 맞아야 한다.
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
ANN_FILE = os.environ.get("TOKENHMR_3DPW_NPZ", "data/tokenhmr/3dpw_test.npz")
PKL = REPO / "data/tokenhmr_3dpw_cuda.pkl"
N_JOINTS, JOINT_DIM = 14, 42

CAPACITY = {
    "64_64": [64, 64],
    "128_128_64": [128, 128, 64],
    "256_256_128": [256, 256, 128],
    "512_512_256": [512, 512, 256],
    "1024_512_256": [1024, 512, 256],
}
LOSSES = ["l1", "l2", "smooth_l1", "huber"]
EPOCHS = [5, 10, 20, 30, 50]
N_SEQS = [5, 10, 15, 23]

BASE_DIMS, BASE_LOSS, BASE_EPOCHS = [128, 128, 64], "l1", 10


class RefinementMLP(nn.Module):
    def __init__(self, hidden_dims):
        super().__init__()
        layers, prev = [], JOINT_DIM
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        layers.append(nn.Linear(prev, JOINT_DIM))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class PairDataset(Dataset):
    def __init__(self, pred, gt):
        self.pred = torch.from_numpy(pred.reshape(-1, JOINT_DIM)).float()
        self.gt = torch.from_numpy(gt.reshape(-1, JOINT_DIM)).float()
        self.residual = self.gt - self.pred

    def __len__(self):
        return len(self.pred)

    def __getitem__(self, i):
        return {"pred": self.pred[i], "gt": self.gt[i], "residual": self.residual[i]}


def similarity_transform(S1, S2):
    B = S1.shape[0]
    S1, S2 = S1.permute(0, 2, 1), S2.permute(0, 2, 1)
    mu1, mu2 = S1.mean(2, keepdim=True), S2.mean(2, keepdim=True)
    X1, X2 = S1 - mu1, S2 - mu2
    var1 = (X1 ** 2).sum(dim=(1, 2))
    K = torch.bmm(X1, X2.permute(0, 2, 1))
    U, _, Vh = torch.linalg.svd(K)
    Z = torch.eye(3, device=S1.device).unsqueeze(0).repeat(B, 1, 1)
    Z[:, -1, -1] = torch.sign(torch.linalg.det(torch.bmm(U, Vh)))
    R = torch.bmm(torch.bmm(Vh.permute(0, 2, 1), Z), U.permute(0, 2, 1))
    trace = torch.bmm(R, K).diagonal(dim1=-2, dim2=-1).sum(-1)
    scale = (trace / var1).unsqueeze(-1).unsqueeze(-1)
    return (scale * torch.bmm(R, S1) + (mu2 - scale * torch.bmm(R, mu1))).permute(0, 2, 1)


def pa_mpjpe(p, g):
    return torch.sqrt(((similarity_transform(p, g) - g) ** 2).sum(-1)).mean(-1) * 1000


def make_criterion(kind):
    return {"l1": nn.L1Loss(), "l2": nn.MSELoss(),
            "smooth_l1": nn.SmoothL1Loss(), "huber": nn.HuberLoss(delta=1.0)}[kind]


def run_loo(pred, gt, seq_to_idx, device, *, hidden_dims, loss_kind, epochs,
            n_train_seqs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    seqs = sorted(seq_to_idx)
    before, after = [], []
    for s in seqs:
        train_seqs = [q for q in seqs if q != s]
        if n_train_seqs is not None and n_train_seqs < len(train_seqs):
            # 시드마다 다른 부분집합을 뽑아야 편차가 학습셋 선택까지 반영된다
            rng = np.random.default_rng(seed * 1000 + seqs.index(s))
            train_seqs = list(rng.choice(train_seqs, n_train_seqs, replace=False))
        tr_i = np.concatenate([seq_to_idx[q] for q in train_seqs])
        te_i = seq_to_idx[s]

        model = RefinementMLP(hidden_dims).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit = make_criterion(loss_kind)
        loader = DataLoader(PairDataset(pred[tr_i], gt[tr_i]), batch_size=256,
                            shuffle=True, num_workers=2)
        for _ in range(epochs):
            model.train()
            for b in loader:
                loss = crit(model(b["pred"].to(device)), b["residual"].to(device))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()

        model.eval()
        with torch.no_grad():
            for b in DataLoader(PairDataset(pred[te_i], gt[te_i]), batch_size=256,
                                num_workers=2):
                p, g = b["pred"].to(device), b["gt"].to(device)
                r = p + model(p)
                before += pa_mpjpe(p.view(-1, N_JOINTS, 3), g.view(-1, N_JOINTS, 3)).cpu().tolist()
                after += pa_mpjpe(r.view(-1, N_JOINTS, 3), g.view(-1, N_JOINTS, 3)).cpu().tolist()

    b, a = float(np.mean(before)), float(np.mean(after))
    return {"before": b, "after": a, "pct": 100 * (b - a) / b}


def sweep(tag, variants, pred, gt, seq_to_idx, device, seeds, results):
    print(f"\n########## {tag}", flush=True)
    for name, kw in variants.items():
        runs = [run_loo(pred, gt, seq_to_idx, device, seed=sd, **kw) for sd in seeds]
        pct = np.array([r["pct"] for r in runs])
        aft = np.array([r["after"] for r in runs])
        key = f"{tag}_{name}"
        results[key] = {"config": {k: str(v) for k, v in kw.items()},
                        "runs": runs, "pct_mean": float(pct.mean()),
                        "pct_std": float(pct.std(ddof=1)),
                        "after_mean": float(aft.mean()),
                        "after_std": float(aft.std(ddof=1))}
        print(f"  {name:14s} {aft.mean():7.3f} +- {aft.std(ddof=1):.3f} mm   "
              f"{pct.mean():6.2f} +- {pct.std(ddof=1):.2f} %   "
              f"@{key}_pct_mean={pct.mean():.4f} @{key}_pct_std={pct.std(ddof=1):.4f}",
              flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seeds", default="42,123,456")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(PKL, "rb") as f:
        s = pickle.load(f)
    pred = np.stack([x["pred_joints_3d_eval"] for x in s])
    gt = np.stack([x["gt_joints_3d"] for x in s])
    imgnames = np.load(ANN_FILE, allow_pickle=True)["imgname"]
    tmp = defaultdict(list)
    for i, nm in enumerate(imgnames):
        tmp[nm.split("/")[1]].append(i)
    seq_to_idx = {k: np.array([i for i in v if i < len(pred)]) for k, v in tmp.items()}
    print(f"TokenHMR {len(pred)} samples, {len(seq_to_idx)} sequences, seeds={seeds}",
          flush=True)
    for n, d in CAPACITY.items():
        print(f"  capacity {n:14s} params={RefinementMLP(d).n_params()}", flush=True)

    results = {}
    base = dict(hidden_dims=BASE_DIMS, loss_kind=BASE_LOSS, epochs=BASE_EPOCHS,
                n_train_seqs=None)

    sweep("cap", {n: {**base, "hidden_dims": d} for n, d in CAPACITY.items()},
          pred, gt, seq_to_idx, device, seeds, results)
    sweep("loss", {l: {**base, "loss_kind": l} for l in LOSSES},
          pred, gt, seq_to_idx, device, seeds, results)
    sweep("ep", {str(e): {**base, "epochs": e} for e in EPOCHS},
          pred, gt, seq_to_idx, device, seeds, results)
    sweep("nseq", {str(k): {**base, "n_train_seqs": (None if k == 23 else k)}
                   for k in N_SEQS},
          pred, gt, seq_to_idx, device, seeds, results)

    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}/results.json", flush=True)


if __name__ == "__main__":
    main()
