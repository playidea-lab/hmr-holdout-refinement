"""행 순서 교정 SPIN 의 라벨 예산 곡선·고정 한 벌·다른 배우 최근접을 다시 잰다.

B 논문의 tab:efficiency / fig:fixedquery / tab:diffsubj SPIN 칸 대체용. 프로토콜은 exp803b·12d1b249·exp774 와 같다:
- 최근접: 대상 시퀀스의 평균 예측 관절(42차원)에 가장 가까운 후보 k개(후보 = 나머지 23개), 학습 시드 0.
- 무작위: 후보 23개 중 k개, 추첨 20회(추첨 d 마다 학습 시드 d).
- 수치는 전체 행 프레임 가중 PA-MPJPE(mm); Wilcoxon 은 24 fold 짝(최근접 vs 무작위 평균).
- 고정 한 벌(k=5): 무작위 10벌·spread(최원점)·medoid(전체 평균 최근접), 한 벌 밖 19 시퀀스에서 평가,
  같은 19 시퀀스에서 대상별 최근접(k=5)과 비교.
- 다른 배우 최근접: 대상 시퀀스의 배우와 겹치는 배우가 없는 후보만, k=1·2.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import exp775_share_seeds as E  # noqa: E402

KS = (1, 2, 3, 5, 10)
N_DRAWS = 20
N_FIXED = 10


def fit_eval(X, Y, tr, te, seed):
    m = E.train(X, Y, tr, seed, 10)
    return E.evaluate(m, X, Y, te)


def main() -> int:
    names, center, seq, subj = E.load_rows()
    p, g = E.load_backbone("SPIN", names, center, spin_reorder=True)
    n = len(p)
    X = torch.as_tensor(p.reshape(n, -1), dtype=torch.float32, device=E.DEV)
    Y = torch.as_tensor(g.reshape(n, -1), dtype=torch.float32, device=E.DEV)
    base = E.pa_torch(X, Y).cpu().numpy()
    seqs = sorted(set(seq.tolist()))
    sidx = {s: np.nonzero(seq == s)[0] for s in seqs}
    mean_pred = {s: p[sidx[s]].reshape(len(sidx[s]), -1).mean(0) for s in seqs}
    subj_of = {s: set(subj[sidx[s]].tolist()) for s in seqs}
    print(f"baseline {base.mean():.3f}", flush=True)

    near = {k: np.zeros(n) for k in KS}
    rand = {k: np.zeros((N_DRAWS, n)) for k in KS}
    other = {k: np.zeros(n) for k in (1, 2)}
    for fi, s in enumerate(seqs):
        te = sidx[s]
        cands = [c for c in seqs if c != s]
        order = sorted(cands, key=lambda c: np.linalg.norm(mean_pred[c] - mean_pred[s]))
        oth = [c for c in order if not (subj_of[c] & subj_of[s])]
        for k in KS:
            tr = np.concatenate([sidx[c] for c in order[:k]])
            near[k][te] = fit_eval(X, Y, tr, te, 0)
            for d in range(N_DRAWS):
                rng = np.random.default_rng(1000 * fi + 10 * k + d)
                pick = rng.choice(len(cands), k, replace=False)
                tr = np.concatenate([sidx[cands[i]] for i in pick])
                rand[k][d, te] = fit_eval(X, Y, tr, te, d)
        for k in (1, 2):
            tr = np.concatenate([sidx[c] for c in oth[:k]])
            other[k][te] = fit_eval(X, Y, tr, te, 0)
        print(f"  fold {fi + 1}/24 {s}", flush=True)

    for k in KS:
        rmm = rand[k].mean(1)
        fold_near = np.array([near[k][sidx[s]].mean() for s in seqs])
        fold_rand = np.array([rand[k][:, sidx[s]].mean() for s in seqs])
        w = stats.wilcoxon(fold_near, fold_rand)
        print(f"@spin_k{k}_near_mm={near[k].mean():.3f} @spin_k{k}_near_pct={E.pct(base, near[k]):.2f} "
              f"@spin_k{k}_rand_mm={rmm.mean():.3f} @spin_k{k}_rand_sd={rmm.std(ddof=1):.3f} "
              f"@spin_k{k}_rand_pct={100 * (base.mean() - rmm.mean()) / base.mean():.2f} "
              f"@spin_k{k}_wilcoxon_p={w.pvalue:.3e}", flush=True)
    for k in (1, 2):
        print(f"@spin_other_k{k}_pct={E.pct(base, other[k]):.2f}", flush=True)

    # 고정 한 벌 (k=5): 한 벌 밖 19 시퀀스에서 평가
    def fixed_eval(S, seed=0):
        tr = np.concatenate([sidx[c] for c in S])
        rest = [c for c in seqs if c not in S]
        te = np.concatenate([sidx[c] for c in rest])
        after = fit_eval(X, Y, tr, te, seed)
        q = near[5][te]
        return E.pct(base[te], after), E.pct(base[te], q)

    rng = np.random.default_rng(7)
    fx, qx = [], []
    for d in range(N_FIXED):
        S = list(rng.choice(seqs, 5, replace=False))
        a, b = fixed_eval(S, d)
        fx.append(a)
        qx.append(b)
    wins = int(sum(b > a for a, b in zip(fx, qx)))
    print(f"@spin_fixed_rand_pct={np.mean(fx):.2f} @spin_fixed_rand_sd={np.std(fx, ddof=1):.2f} "
          f"@spin_fixed_query_pct={np.mean(qx):.2f} @spin_fixed_query_sd={np.std(qx, ddof=1):.2f} "
          f"@spin_query_wins={wins}", flush=True)
    M = np.stack([mean_pred[s] for s in seqs])
    overall = M.mean(0)
    medoid = [seqs[i] for i in np.argsort(np.linalg.norm(M - overall, axis=1))[:5]]
    spread = [seqs[int(np.argmax(np.linalg.norm(M - overall, axis=1)))]]
    while len(spread) < 5:
        dmin = np.min(np.stack([np.linalg.norm(M - mean_pred[c], axis=1) for c in spread]), 0)
        for c in spread:
            dmin[seqs.index(c)] = -1
        spread.append(seqs[int(np.argmax(dmin))])
    for tag, S in (("medoid", medoid), ("spread", spread)):
        a, b = fixed_eval(S)
        print(f"@spin_fixed_{tag}_pct={a:.2f} @spin_fixed_{tag}_query_pct={b:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
