"""고정된 다섯 시퀀스 한 벌 대 테스트마다 다시 고르는 최근접 선택.

**왜**: IEEE Access 리뷰어 1의 지적 — "다섯 시퀀스를 테스트 시퀀스마다 다시 고르므로 매번
다른 다섯 개가 쓰인다. 고정된 한 벌이 모든 테스트에서 어떻게 작동하는지 보여달라."
논문에 그 값이 없다.

**설계**: 고정 한 벌 S (|S|=5) 를 고르고 한 번만 학습한다. 평가는 S 에 없는 19개 시퀀스에서
한다(S 안의 시퀀스를 평가에 쓰면 누출이다). **비교 공정성을 위해 최근접 선택도 같은 19개에서만
평가한다** — 그러지 않으면 테스트 집합이 달라 비교가 안 된다.

**고정 한 벌을 고르는 세 방식** (모두 GT 를 쓰지 않는다 — 실무에서 가능해야 한다):
  random   무작위 n_trials 회, 평균과 표준편차
  spread   예측 공간에서 서로 가장 멀게 (greedy farthest-point) — 커버리지 직관
  medoid   전체 평균 예측에 가장 가까운 다섯 — 대표성 직관

**무엇이 걸리나**: 고정 한 벌이 최근접보다 확실히 나쁘면 "예산보다 선택이 중요하다"는 주장의
직접 증거다. 비슷하면 그 주장을 약화시켜야 한다. 어느 쪽이든 리뷰어 질문에 답한다.

모델·학습·PA 규약은 exp803b 에서 그대로 import 한다(표준 Procrustes).
"""

import os
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from exp803b_data_efficiency_stats import (  # noqa: E402
    BACKBONES, MOCAP, compute_pa_mpjpe, eval_mlp, load_backbone, train_mlp,
)

import torch  # noqa: E402  (exp803b 가 먼저 torch 를 설정한다)

K = 5
OUT = Path(os.path.join(os.environ.get("RESULTS_DIR", "outputs"), "exp773"))


def pick_spread(seqs, mean_pred, k):
    """greedy farthest-point: 전체 중심에서 가장 먼 것부터, 이후 선택된 것들과 최대로 멀게."""
    c = np.mean([mean_pred[s] for s in seqs], axis=0)
    first = max(seqs, key=lambda s: np.linalg.norm(mean_pred[s] - c))
    chosen = [first]
    while len(chosen) < k:
        nxt = max((s for s in seqs if s not in chosen),
                  key=lambda s: min(np.linalg.norm(mean_pred[s] - mean_pred[q])
                                    for q in chosen))
        chosen.append(nxt)
    return chosen


def pick_medoid(seqs, mean_pred, k):
    """전체 평균 예측에 가장 가까운 k 개."""
    c = np.mean([mean_pred[s] for s in seqs], axis=0)
    return sorted(seqs, key=lambda s: np.linalg.norm(mean_pred[s] - c))[:k]


def evaluate_fixed(train_seqs, test_seqs, preds, gts, seq_idx, dev, epochs, seed):
    """고정 한 벌로 한 번 학습하고 test_seqs 전체를 평가한다 (프레임 가중)."""
    tr = np.concatenate([seq_idx[s] for s in train_seqs])
    m = train_mlp(preds[tr], gts[tr], dev, epochs, seed=seed)
    te = np.concatenate([seq_idx[s] for s in test_seqs])
    return float(eval_mlp(m, preds[te], gts[te], dev).mean())


def evaluate_nearest(test_seqs, all_seqs, preds, gts, seq_idx, mean_pred, dev,
                     epochs, seed):
    """테스트 시퀀스마다 최근접 k 개를 다시 고른다 — 논문의 방식."""
    vals, ns = [], []
    for s in test_seqs:
        cands = [q for q in all_seqs if q != s]
        d = np.array([np.linalg.norm(mean_pred[s] - mean_pred[q]) for q in cands])
        pick = [cands[i] for i in np.argsort(d)[:K]]
        tr = np.concatenate([seq_idx[q] for q in pick])
        m = train_mlp(preds[tr], gts[tr], dev, epochs, seed=seed)
        v = eval_mlp(m, preds[seq_idx[s]], gts[seq_idx[s]], dev)
        vals.append(v.sum())
        ns.append(len(v))
    return float(np.sum(vals) / np.sum(ns))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="TokenHMR,4DHumans")
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mocap = np.load(MOCAP, allow_pickle=True)
    imgnames = [s.decode() if isinstance(s, bytes) else s for s in mocap["imgname"]]
    seq_names = [Path(n).parts[-2] if len(Path(n).parts) >= 2 else "unknown" for n in imgnames]
    print(f"device={dev}  k={K}  시행 {a.n_trials}회", flush=True)

    out, t0 = {}, time.time()
    for bb in [x.strip() for x in a.backbones.split(",")]:
        preds, gts, sn = load_backbone(BACKBONES[bb], seq_names)
        seqs = sorted(set(sn.tolist()))
        seq_idx = {s: np.nonzero(sn == s)[0] for s in seqs}
        mean_pred = {s: preds[seq_idx[s]].reshape(len(seq_idx[s]), -1).mean(0) for s in seqs}
        print(f"\n===== {bb}  시퀀스 {len(seqs)}", flush=True)

        row = {}
        rng = np.random.default_rng(a.seed)
        # --- 무작위 고정 한 벌 ---
        vals, tests = [], None
        for t in range(a.n_trials):
            S = sorted(rng.choice(seqs, K, replace=False).tolist())
            T = [s for s in seqs if s not in S]
            v = evaluate_fixed(S, T, preds, gts, seq_idx, dev, a.epochs, a.seed)
            base = float(compute_pa_mpjpe(
                preds[np.concatenate([seq_idx[s] for s in T])],
                gts[np.concatenate([seq_idx[s] for s in T])]).mean())
            near = evaluate_nearest(T, seqs, preds, gts, seq_idx, mean_pred, dev,
                                    a.epochs, a.seed)
            vals.append({"S": S, "fixed": v, "nearest": near, "baseline": base,
                         "n_test": len(T)})
            print(f"  무작위 {t+1:2d}/{a.n_trials}  고정 {v:6.3f}  최근접 {near:6.3f}  "
                  f"기준 {base:6.3f}  ({(time.time()-t0)/60:.1f} min)", flush=True)
        row["random"] = vals
        fx = np.array([v["fixed"] for v in vals])
        nr = np.array([v["nearest"] for v in vals])
        bs = np.array([v["baseline"] for v in vals])
        print(f"  [무작위 고정] {fx.mean():.3f} ±{fx.std(ddof=1):.3f}  |  "
              f"[같은 테스트의 최근접] {nr.mean():.3f} ±{nr.std(ddof=1):.3f}  |  "
              f"기준 {bs.mean():.3f}", flush=True)
        print(f"  개선  고정 {100*(bs.mean()-fx.mean())/bs.mean():+.2f}%   "
              f"최근접 {100*(bs.mean()-nr.mean())/bs.mean():+.2f}%", flush=True)

        # --- 규칙으로 고른 고정 한 벌 ---
        for name, fn in (("spread", pick_spread), ("medoid", pick_medoid)):
            S = sorted(fn(seqs, mean_pred, K))
            T = [s for s in seqs if s not in S]
            idxT = np.concatenate([seq_idx[s] for s in T])
            base = float(compute_pa_mpjpe(preds[idxT], gts[idxT]).mean())
            v = evaluate_fixed(S, T, preds, gts, seq_idx, dev, a.epochs, a.seed)
            near = evaluate_nearest(T, seqs, preds, gts, seq_idx, mean_pred, dev,
                                    a.epochs, a.seed)
            row[name] = {"S": S, "fixed": v, "nearest": near, "baseline": base}
            print(f"  [{name}] {v:.3f} ({100*(base-v)/base:+.2f}%)  |  "
                  f"최근접 {near:.3f} ({100*(base-near)/base:+.2f}%)  |  기준 {base:.3f}",
                  flush=True)
            print(f"    고른 것: {S}", flush=True)

        print(f"  @{bb}_fixed_random_pa={fx.mean():.4f} @{bb}_nearest_pa={nr.mean():.4f} "
              f"@{bb}_fixed_spread_pa={row['spread']['fixed']:.4f} "
              f"@{bb}_fixed_medoid_pa={row['medoid']['fixed']:.4f} "
              f"@{bb}_baseline_pa={bs.mean():.4f}", flush=True)
        out[bb] = row

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(
        {"k": K, "n_trials": a.n_trials, "epochs": a.epochs,
         "note": "고정 한 벌은 S 밖의 시퀀스에서만 평가하고, 최근접도 같은 테스트 집합에서 평가",
         "backbones": out}, indent=2))
    print(f"\n저장: {OUT/'results.json'}  (총 {(time.time()-t0)/60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
