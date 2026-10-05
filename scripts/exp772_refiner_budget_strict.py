"""정제기(관절 공간)의 라벨 예산 곡선을 헤드 적응과 **같은 엄격한 누출 경계**로 잰다.

**왜 다시 재는가**: 논문 B 의 §subjeff 는 피험자를 **행 단위**로 갈랐다. 3DPW 의 24개 시퀀스 중
13개에 두 사람이 주석돼 있어서, 피험자 u 를 보류해도 같은 시퀀스의 다른 사람 행은 학습에 쓸 수
있다. 정제기는 주석된 사람의 관절 42개만 보므로 그 경계가 정당하다.

그런데 헤드 적응(exp771)은 크롭의 **이미지 특징**을 읽는다. 그 크롭에 보류 피험자가 찍혀
있으면 이미지를 통해 본 것이므로, exp771 은 보류 피험자의 프레임이 하나라도 있는 시퀀스를
후보에서 전부 뺐다(시퀀스 단위). 그 결과 k 상한이 4 로 잘렸다.

**두 곡선을 나란히 놓으려면 같은 저울이어야 한다.** 느슨한 경계의 정제기와 엄격한 경계의 헤드를
비교하면 헤드에 불리하다. 그래서 정제기를 exp771 과 동일한 규약으로 다시 잰다 —
같은 피험자 정의(betas), 같은 후보 제한(시퀀스 단위 배제), 같은 질의(보류 피험자 전체의 평균
예측), 같은 k.

모델과 학습은 exp803b 와 한 글자도 다르지 않아야 한다. LayerNorm·Dropout·출력층 zero-init·
AdamW weight_decay·gradient clipping 중 하나만 빠져도 오라클이 어긋난다(실측 확인됨).
"""

import os
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

N_JOINTS, N_COORD, DIM, MM = 14, 3, 42, 1000.0
PKL = os.path.join(os.environ.get("PRED_DIR", "data"), "4dhumans_3dpw_cuda.pkl")
ANN = os.environ.get("TOKENHMR_3DPW_NPZ", "data/tokenhmr/3dpw_test.npz")
OUT = Path(os.path.join(os.environ.get("RESULTS_DIR", "outputs"), "exp772"))


class RefineMLP(nn.Module):
    """exp803b 의 RefineMLP 와 동일. 42 -> 128 -> 128 -> 64 -> 42, residual."""

    def __init__(self, input_dim=DIM, hidden_dims=(128, 128, 64), output_dim=DIM):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


def pa_mpjpe(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """표준 Procrustes — 최소제곱 스케일 포함. 논문 전체와 동일 규약."""
    P0 = pred.reshape(-1, N_JOINTS, N_COORD)
    G0 = gt.reshape(-1, N_JOINTS, N_COORD)
    g_mean = G0.mean(1, keepdims=True)
    P, G = P0 - P0.mean(1, keepdims=True), G0 - g_mean
    sp = np.sqrt((P ** 2).sum((1, 2)))[:, None, None]
    sg = np.sqrt((G ** 2).sum((1, 2)))[:, None, None]
    Pn, Gn = P / (sp + 1e-8), G / (sg + 1e-8)
    H = np.einsum("nji,njk->nik", Pn, Gn)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("nij,njk->nik",
                                        Vt.transpose(0, 2, 1), U.transpose(0, 2, 1))))
    D = np.zeros((len(P), 3, 3))
    D[:, 0, 0] = D[:, 1, 1] = 1.0
    D[:, 2, 2] = d
    R = np.einsum("nij,njk,nkl->nil", Vt.transpose(0, 2, 1), D, U.transpose(0, 2, 1))
    scale = S[:, 0] + S[:, 1] + d * S[:, 2]
    aligned = sg * scale[:, None, None] * np.einsum("nij,nkj->nki", R, Pn) + g_mean
    return np.sqrt(((aligned - G0) ** 2).sum(-1)).mean(-1) * MM


def train_mlp(preds, gts, dev, epochs=10, lr=1e-3, batch=256, seed=0):
    torch.manual_seed(seed)
    model = RefineMLP().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    x = torch.tensor(preds.reshape(-1, DIM), dtype=torch.float32, device=dev)
    y = torch.tensor(gts.reshape(-1, DIM), dtype=torch.float32, device=dev)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y),
                                         batch_size=batch, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            nn.functional.l1_loss(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    return model


@torch.no_grad()
def refine(model, preds, dev) -> np.ndarray:
    model.eval()
    x = torch.tensor(preds.reshape(-1, DIM), dtype=torch.float32, device=dev)
    return model(x).cpu().numpy().reshape(-1, N_JOINTS, N_COORD)


def build_folds():
    """exp771·exp760 과 동일한 피험자 정의와 후보 제한."""
    with open(PKL, "rb") as f:
        data = pickle.load(f)
    preds = np.array([s["pred_joints_3d_eval"].reshape(N_JOINTS, N_COORD) for s in data])
    gts = np.array([s["gt_joints_3d"].reshape(N_JOINTS, N_COORD) for s in data])
    n = len(preds)

    ann = np.load(ANN, allow_pickle=True)
    imgnames, betas = ann["imgname"][:n], ann["betas"][:n]
    lut, subj = {}, []
    for b in betas:
        subj.append(lut.setdefault(tuple(np.round(b, 3)), len(lut)))
    subj = np.array(subj)

    seq_of = np.array([str(s).split("/")[1] for s in imgnames])
    seqs = sorted(set(seq_of.tolist()))
    seq_idx = {s: np.nonzero(seq_of == s)[0] for s in seqs}

    folds = {}
    for u in sorted(set(subj.tolist())):
        te = np.nonzero(subj == u)[0]
        # 보류 피험자의 프레임이 하나도 없는 시퀀스만 후보 — exp771 과 동일한 엄격 경계
        cands = [s for s in seqs if not (subj[seq_idx[s]] == u).any()]
        folds[u] = {"test": te, "cands": cands}
    return preds, gts, seqs, seq_idx, folds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", default="1,2,3,4")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-oracle", action="store_true",
                    help="fold 별 후보 풀 전체로 학습한 천장도 함께 낸다")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds, gts, seqs, seq_idx, folds = build_folds()
    print(f"device={dev}  n={len(preds)}  시퀀스 {len(seqs)}  피험자 {len(folds)}", flush=True)

    pool_min = min(len(f["cands"]) for f in folds.values())
    for u, f in folds.items():
        print(f"  subject_{u}: 테스트 {len(f['test']):6d}  후보 시퀀스 {len(f['cands']):3d}",
              flush=True)
    print(f"최소 후보 시퀀스 = {pool_min}  @pool_min={pool_min}", flush=True)

    ks = [int(x) for x in a.k.split(",") if int(x) <= pool_min]
    if a.with_oracle:
        ks.append(-1)                                  # -1 = 후보 풀 전체
    print(f"예산 스윕 k={ks} (-1 은 fold 별 전체 풀)\n", flush=True)

    # 미적응 baseline 은 k 와 무관 — fold 당 한 번만
    base = {u: pa_mpjpe(preds[f["test"]], gts[f["test"]]) for u, f in folds.items()}
    seq_mean = {s: preds[seq_idx[s]].reshape(len(seq_idx[s]), -1).mean(0) for s in seqs}

    results, t0 = {}, time.time()
    for k in ks:
        before_all, after_all, per_fold = [], [], {}
        for u, f in folds.items():
            te = f["test"]
            q = preds[te].reshape(len(te), -1).mean(0)     # 질의 = 보류 피험자 전체 평균 예측
            d = np.array([np.linalg.norm(q - seq_mean[s]) for s in f["cands"]])
            order = np.argsort(d)
            pick = [f["cands"][i] for i in (order if k == -1 else order[:k])]
            tr = np.concatenate([seq_idx[s] for s in pick])

            m = train_mlp(preds[tr], gts[tr], dev, a.epochs, seed=a.seed)
            after = pa_mpjpe(refine(m, preds[te], dev), gts[te])
            per_fold[f"subject_{u}"] = {
                "n": int(len(te)), "n_picked": len(pick),
                "before": float(base[u].mean()), "after": float(after.mean())}
            before_all += base[u].tolist()
            after_all += after.tolist()
            print(f"  k={k:>2} subject_{u}: {base[u].mean():6.2f} -> {after.mean():6.2f} "
                  f"({100*(base[u].mean()-after.mean())/base[u].mean():+.2f}%)  "
                  f"시퀀스 {len(pick)}개  ({(time.time()-t0)/60:.1f} min)", flush=True)

        b, aft = float(np.mean(before_all)), float(np.mean(after_all))
        fb = float(np.mean([v["before"] for v in per_fold.values()]))
        fa = float(np.mean([v["after"] for v in per_fold.values()]))
        label = "all" if k == -1 else str(k)
        results[label] = {
            "before_pa": b, "after_pa": aft, "imp_pct": 100 * (b - aft) / b,
            "subjw_before_pa": fb, "subjw_after_pa": fa,
            "subjw_imp_pct": 100 * (fb - fa) / fb, "per_fold": per_fold}
        print(f"k={label}  프레임가중 {b:.4f} -> {aft:.4f} ({100*(b-aft)/b:+.2f}%)  |  "
              f"피험자가중 {fb:.4f} -> {fa:.4f} ({100*(fb-fa)/fb:+.2f}%)", flush=True)
        print(f"  @k{label}_after_pa={aft:.4f} @k{label}_imp_pct={100*(b-aft)/b:.4f}\n",
              flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(
        {"pool_min": pool_min, "n_folds": len(folds), "epochs": a.epochs,
         "boundary": "sequence-level subject exclusion (matches exp771)",
         "k": results}, indent=2))
    print(f"저장: {OUT/'results.json'}  (총 {(time.time()-t0)/60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
