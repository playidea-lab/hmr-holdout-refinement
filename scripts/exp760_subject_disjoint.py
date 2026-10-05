#!/usr/bin/env python3
"""exp760: 배우를 분리한 LOO와 시퀀스 LOO를 같은 조건에서 나란히 잰다.

3DPW test의 배우는 betas 기준 5명뿐이고, 배우 1은 24개 중 20개 시퀀스에,
배우 0은 14개에 나온다. 시퀀스 LOO는 22/24 폴드에서 held-out 시퀀스의 배우를
학습셋에 그대로 남긴다 — 즉 refiner가 그 체형을 이미 봤다.

여기서는 같은 모델·같은 학습 설정으로 두 프로토콜을 돌려 차이를 본다.
차이가 작으면 논문의 ceiling은 배우 누출에 기대지 않는 것이고,
크면 보고된 개선의 상당 부분이 체형 재보정이었다는 뜻이다.
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
BACKBONE_PATHS = {
    "TokenHMR": REPO / "data/tokenhmr_3dpw_cuda.pkl",
    "4DHumans": REPO / "data/4dhumans_3dpw_cuda.pkl",
    "SPIN": REPO / "data/spin_3dpw_14j.pkl",
}
N_JOINTS = 14
JOINT_DIM = N_JOINTS * 3


class RefinementMLP(nn.Module):
    def __init__(self, input_dim=JOINT_DIM, hidden_dims=None, output_dim=JOINT_DIM):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128, 64]
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


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
    mu1, mu2 = S1.mean(dim=2, keepdim=True), S2.mean(dim=2, keepdim=True)
    X1, X2 = S1 - mu1, S2 - mu2
    var1 = (X1 ** 2).sum(dim=(1, 2))
    K = torch.bmm(X1, X2.permute(0, 2, 1))
    U, _, Vh = torch.linalg.svd(K)
    V = Vh.permute(0, 2, 1)
    Z = torch.eye(3, device=S1.device).unsqueeze(0).repeat(B, 1, 1)
    Z[:, -1, -1] = torch.sign(torch.linalg.det(torch.bmm(U, Vh)))
    R = torch.bmm(torch.bmm(V, Z), U.permute(0, 2, 1))
    trace = torch.bmm(R, K).diagonal(dim1=-2, dim2=-1).sum(-1)
    scale = (trace / var1).unsqueeze(-1).unsqueeze(-1)
    t = mu2 - scale * torch.bmm(R, mu1)
    return (scale * torch.bmm(R, S1) + t).permute(0, 2, 1)


def pa_mpjpe(p, g):
    return torch.sqrt(((similarity_transform(p, g) - g) ** 2).sum(-1)).mean(-1) * 1000


def train_model(loader, device, epochs):
    m = RefinementMLP().to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.L1Loss()
    for _ in range(epochs):
        m.train()
        for b in loader:
            loss = crit(m(b["pred"].to(device)), b["residual"].to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        sch.step()
    return m


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    before, after = [], []
    for b in loader:
        pred, gt = b["pred"].to(device), b["gt"].to(device)
        ref = pred + model(pred)
        p, r, g = (t.view(-1, N_JOINTS, 3) for t in (pred, ref, gt))
        before += pa_mpjpe(p, g).cpu().tolist()
        after += pa_mpjpe(r, g).cpu().tolist()
    return np.array(before), np.array(after)


def run_folds(pred, gt, folds, device, epochs):
    """folds: {이름: held-out 행 인덱스}. 학습은 나머지 전부."""
    all_b, all_a, per_fold = [], [], {}
    n = len(pred)
    for name, test_idx in folds.items():
        test_idx = np.asarray(test_idx)
        mask = np.ones(n, dtype=bool)
        mask[test_idx] = False
        train_idx = np.nonzero(mask)[0]
        tr = PairDataset(pred[train_idx], gt[train_idx])
        te = PairDataset(pred[test_idx], gt[test_idx])
        model = train_model(DataLoader(tr, batch_size=256, shuffle=True, num_workers=2),
                            device, epochs)
        b, a = evaluate(model, DataLoader(te, batch_size=256, num_workers=2), device)
        all_b.append(b)
        all_a.append(a)
        per_fold[str(name)] = {"n": int(len(b)), "before": float(b.mean()),
                               "after": float(a.mean()),
                               "pct": float(100 * (b.mean() - a.mean()) / b.mean())}
    b, a = np.concatenate(all_b), np.concatenate(all_a)
    return {
        "before": float(b.mean()), "after": float(a.mean()),
        "pct": float(100 * (b.mean() - a.mean()) / b.mean()),
        "n_folds_improved": sum(1 for v in per_fold.values() if v["after"] < v["before"]),
        "n_folds": len(per_fold), "per_fold": per_fold,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ann = np.load(ANN_FILE, allow_pickle=True)
    imgnames, betas = ann["imgname"], ann["betas"]

    # betas가 곧 배우 신원이다
    lut, subj = {}, []
    for b in betas:
        k = tuple(np.round(b, 3))
        subj.append(lut.setdefault(k, len(lut)))
    subj = np.array(subj)

    seq_of = np.array([n.split("/")[1] for n in imgnames])
    seq_folds = {s: np.nonzero(seq_of == s)[0] for s in sorted(set(seq_of.tolist()))}
    subj_folds = {f"subject_{u}": np.nonzero(subj == u)[0] for u in sorted(set(subj.tolist()))}
    print(f"시퀀스 폴드 {len(seq_folds)}개, 배우 폴드 {len(subj_folds)}개 "
          f"(행 수 {[len(v) for v in subj_folds.values()]})", flush=True)

    results = {}
    for name, path in BACKBONE_PATHS.items():
        with open(path, "rb") as f:
            s = pickle.load(f)
        pred = np.stack([x["pred_joints_3d_eval"] for x in s])
        gt = np.stack([x["gt_joints_3d"] for x in s])
        n = min(len(pred), len(subj))
        pred, gt = pred[:n], gt[:n]
        sf = {k: v[v < n] for k, v in seq_folds.items()}
        uf = {k: v[v < n] for k, v in subj_folds.items()}

        seq_r = run_folds(pred, gt, sf, device, args.epochs)
        subj_r = run_folds(pred, gt, uf, device, args.epochs)
        results[name] = {"sequence_loo": seq_r, "subject_loo": subj_r}

        print(f"\n===== {name} =====", flush=True)
        print(f"  시퀀스 LOO : {seq_r['before']:.2f} -> {seq_r['after']:.2f} "
              f"({seq_r['pct']:+.2f}%)  개선 폴드 "
              f"{seq_r['n_folds_improved']}/{seq_r['n_folds']}", flush=True)
        print(f"  배우   LOO : {subj_r['before']:.2f} -> {subj_r['after']:.2f} "
              f"({subj_r['pct']:+.2f}%)  개선 폴드 "
              f"{subj_r['n_folds_improved']}/{subj_r['n_folds']}", flush=True)
        for k, v in subj_r["per_fold"].items():
            print(f"      {k:12s} n={v['n']:6d}  {v['before']:7.2f} -> "
                  f"{v['after']:7.2f}  ({v['pct']:+.2f}%)", flush=True)
        print(f"  @{name}_seq_pa_pct={seq_r['pct']:.4f} "
              f"@{name}_subj_pa_pct={subj_r['pct']:.4f} "
              f"@{name}_subj_pa_after={subj_r['after']:.4f}", flush=True)

    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}/results.json", flush=True)


if __name__ == "__main__":
    main()
