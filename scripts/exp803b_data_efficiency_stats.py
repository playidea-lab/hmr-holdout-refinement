"""데이터 효율 실험의 통계를 단단히 한다 — 리뷰 주요의견 2에 대한 답.

원본 exp803_data_efficiency.py 의 세 가지를 고친다.
  1. n_random_trials = 3 → 20 (리뷰어: "k=1은 23개 중 1개라 분산이 크다, 10회 이상 + CI")
  2. `RandomState(42)` 를 fold 루프 **안에서** 만들어 모든 fold가 같은 난수열을 썼다
     → (backbone, k, fold) 별로 독립 시드를 파생시킨다
  3. fold 안에서 trial 을 평균해버려 분산이 사라졌다 → 시행별 값을 끝까지 보존한다
그리고 리뷰어가 요구한 nearest vs random 유의성 검정을 붙인다(시퀀스 단위 대응 Wilcoxon).

모델·학습·PA-MPJPE 는 원본 그대로다. 바뀌는 것은 표집과 집계뿐이다.

CQ_PARAMS: n_trials, k_values, backbones, epochs, seed, out
"""

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

N_JOINTS, N_COORD = 14, 3
DIM = N_JOINTS * N_COORD
MM = 1000.0
CI_Z = 1.96
BACKBONES = {
    "TokenHMR": os.path.join(os.environ.get("PRED_DIR", "data"), "tokenhmr_3dpw_cuda.pkl"),
    "4DHumans": os.path.join(os.environ.get("PRED_DIR", "data"), "4dhumans_3dpw_cuda.pkl"),
    "SPIN": os.path.join(os.environ.get("PRED_DIR", "data"), "spin_3dpw_14j.pkl"),
}
MOCAP = os.environ.get("TOKENHMR_3DPW_NPZ", "data/tokenhmr/3dpw_test.npz")
DEFAULTS = dict(n_trials=20, k_values="1,2,3,5,10,23", epochs=10, seed=0,
                out=os.path.join(os.environ.get("RESULTS_DIR", "outputs"), "exp803b_data_efficiency_stats.json"))


class RefineMLP(nn.Module):
    """원본 exp803 의 RefinementMLP 와 동일해야 한다.

    LayerNorm·Dropout·출력층 zero-init 을 빠뜨리면 오라클이 41.10mm(16.98%)로
    논문의 20.03% 와 어긋난다 — 실측으로 확인했다. 표집만 바꾸는 실험이므로
    모델은 한 글자도 달라지면 안 된다.
    """

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


def compute_pa_mpjpe(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Procrustes 정렬 후 관절 평균 거리(mm). 표준(최소제곱 최적 스케일) 규약.

    exp800/exp803 이래 써 온 판본은 예측을 GT 의 Frobenius 노름에 맞추느라
    최적 스케일 항 trace(Sigma) 를 빠뜨렸다. trace(Sigma) <= 1 이므로 늘 과대
    스케일이 되어 PA 가 체계적으로 커진다(백본별 +0.39~+0.62mm).

    항을 되살리면 백본 pkl 에 저장된 공식 pa_mpjpe 와 소수 4자리까지 일치한다
    (TokenHMR 49.1241 / 4DHumans 54.3258 / SPIN 57.6480, 차 0.0000) — 즉 이것이
    백본 저자들이 보고한 값과 같은 규약이다.
    """
    out = []
    for p, g in zip(pred.reshape(-1, N_JOINTS, N_COORD), gt.reshape(-1, N_JOINTS, N_COORD)):
        mu_p, mu_g = p.mean(0), g.mean(0)
        pc, gc = p - mu_p, g - mu_g
        sp, sg = np.sqrt((pc ** 2).sum()), np.sqrt((gc ** 2).sum())
        pc, gc = pc / (sp + 1e-8), gc / (sg + 1e-8)
        U, S, Vt = np.linalg.svd(pc.T @ gc)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1, 1, d]) @ U.T
        scale = S[0] + S[1] + d * S[2]                    # trace(Sigma), 반사 보정 포함
        aligned = sg * scale * (pc @ R.T) + mu_g
        out.append(np.sqrt(((aligned - g) ** 2).sum(-1)).mean() * MM)
    return np.array(out)


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
def eval_mlp(model, preds, gts, dev) -> np.ndarray:
    model.eval()
    x = torch.tensor(preds.reshape(-1, DIM), dtype=torch.float32, device=dev)
    return compute_pa_mpjpe(model(x).cpu().numpy(), gts)


def load_backbone(pkl_path: str, seq_names: list):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    preds = np.array([s["pred_joints_3d_eval"].reshape(N_JOINTS, N_COORD) for s in data])
    gts = np.array([s["gt_joints_3d"].reshape(N_JOINTS, N_COORD) for s in data])
    n = min(len(preds), len(seq_names))
    return preds[:n], gts[:n], np.array(seq_names[:n])


def summarize(nearest_by_seq: dict, random_by_seq: dict, baseline: float, oracle: float,
              nframes_by_seq: dict = None) -> dict:
    """시퀀스 단위 대응 통계. 시행 분산을 끝까지 보존한다.

    nearest_by_seq: seq -> float (그 fold 의 평균 PA)
    random_by_seq:  seq -> (n_trials,) 배열
    nframes_by_seq: seq -> int. 주면 프레임 가중 평균도 함께 낸다.

    원본 exp803 은 nearest 를 프레임 가중(모든 프레임을 모아 평균)으로, random 을
    시퀀스 가중(fold 평균의 평균)으로 재서 두 팔을 다른 저울에 올렸다. 여기서는
    두 가중을 양쪽에 똑같이 적용해 둘 다 보고한다. 논문 표와 비교할 값은 프레임 가중이다.
    """
    seqs = sorted(nearest_by_seq)
    near = np.array([nearest_by_seq[s] for s in seqs])                 # (S,)
    rand = np.stack([random_by_seq[s] for s in seqs])                  # (S, T)

    # 시행별로 전체 평균을 내면 T 개의 독립 추정치가 남는다 → 이것이 CI 의 근거
    per_trial = rand.mean(0)                                           # (T,)
    sd = per_trial.std(ddof=1)
    half = CI_Z * sd / np.sqrt(len(per_trial))
    rand_mean, near_mean = per_trial.mean(), near.mean()

    frame_weighted = {}
    if nframes_by_seq:
        w = np.array([nframes_by_seq[s] for s in seqs], dtype=float)
        w = w / w.sum()
        fw_trial = (rand * w[:, None]).sum(0)                          # (T,)
        fw_sd = fw_trial.std(ddof=1)
        fw_half = CI_Z * fw_sd / np.sqrt(len(fw_trial))
        frame_weighted = {
            "fw_nearest_mm": float((near * w).sum()),
            "fw_random_mm": float(fw_trial.mean()),
            "fw_random_sd_mm": float(fw_sd),
            "fw_random_ci95_mm": [float(fw_trial.mean() - fw_half), float(fw_trial.mean() + fw_half)],
        }

    # 대응 검정: fold 마다 nearest vs 그 fold 의 random 평균
    paired_rand = rand.mean(1)
    try:
        w_stat, w_p = stats.wilcoxon(near, paired_rand)
    except ValueError:                                                 # 차이가 전부 0
        w_stat, w_p = float("nan"), 1.0
    t_stat, t_p = stats.ttest_rel(near, paired_rand)

    def imp(v):
        return 100.0 * (baseline - v) / baseline

    denom = baseline - oracle
    frac = (lambda v: float(100.0 * (baseline - v) / denom)) if denom > 0 else (lambda v: None)
    for key in ("fw_nearest_mm", "fw_random_mm"):
        if key in frame_weighted:
            frame_weighted[key.replace("_mm", "_oracle_frac_pct")] = frac(frame_weighted[key])
            frame_weighted[key.replace("_mm", "_imp_pct")] = float(
                100.0 * (baseline - frame_weighted[key]) / baseline)
    return {
        **frame_weighted,
        "nearest_mm": float(near_mean),
        "nearest_imp_pct": float(imp(near_mean)),
        "nearest_oracle_frac_pct": frac(near_mean),
        "random_mm": float(rand_mean),
        "random_sd_mm": float(sd),
        "random_ci95_mm": [float(rand_mean - half), float(rand_mean + half)],
        "random_imp_pct": float(imp(rand_mean)),
        "random_imp_ci95_pct": [float(imp(rand_mean + half)), float(imp(rand_mean - half))],
        "random_oracle_frac_pct": frac(rand_mean),
        "n_folds": len(seqs), "n_trials": rand.shape[1],
        "wilcoxon_stat": float(w_stat), "wilcoxon_p": float(w_p),
        "ttest_t": float(t_stat), "ttest_p": float(t_p),
        "folds_nearest_better": int((near < paired_rand).sum()),
        "random_draws_beating_nearest_pct": float(100.0 * (rand < near[:, None]).mean()),
    }


def main() -> int:
    p = {**DEFAULTS, **json.loads(os.environ.get("CQ_PARAMS") or "{}")}
    ks = [int(x) for x in str(p["k_values"]).split(",")]
    n_trials, base_seed = int(p["n_trials"]), int(p["seed"])
    names = [n.strip() for n in p["backbones"].split(",")] if "backbones" in p else list(BACKBONES)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev} params={p}", flush=True)

    mocap = np.load(MOCAP, allow_pickle=True)
    imgnames = [s.decode() if isinstance(s, bytes) else s for s in mocap["imgname"]]
    seq_names = [Path(n).parts[-2] if len(Path(n).parts) >= 2 else "unknown" for n in imgnames]

    results, t0 = {}, time.time()
    for bi, bb in enumerate(names):
        preds, gts, sn = load_backbone(BACKBONES[bb], seq_names)
        seqs = sorted(set(sn.tolist()))
        baseline = float(compute_pa_mpjpe(preds, gts).mean())
        print(f"\n===== {bb} =====  n={len(preds)} 시퀀스 {len(seqs)}  baseline {baseline:.4f} mm", flush=True)

        # 오라클(전체 23개 시퀀스 LOO) — k 곡선의 분모
        oracle_pa = []
        for ts in seqs:
            tr, te = sn != ts, sn == ts
            m = train_mlp(preds[tr], gts[tr], dev, int(p["epochs"]), seed=base_seed)
            oracle_pa.extend(eval_mlp(m, preds[te], gts[te], dev).tolist())
        oracle = float(np.mean(oracle_pa))
        print(f"오라클(23 시퀀스 LOO) {oracle:.4f} mm  ({100 * (baseline - oracle) / baseline:.2f}% 개선)  "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        print(f"@{bb}_baseline={baseline:.4f} @{bb}_oracle={oracle:.4f}", flush=True)
        results[bb] = {"baseline_mm": baseline, "oracle_mm": oracle, "k": {}}

        # 시퀀스별 평균 예측 — nearest 선택의 거리 기준 (원본과 동일: 42D L2)
        centroid = {s: preds[sn == s].reshape(-1, DIM).mean(0) for s in seqs}
        nframes = {s: int((sn == s).sum()) for s in seqs}

        for ki, k in enumerate(ks):
            near_by_seq, rand_by_seq, skip = {}, {}, False
            for ts in seqs:
                cands = [s for s in seqs if s != ts]
                if k >= len(cands):
                    skip = True
                    break
                te = sn == ts
                order = sorted(cands, key=lambda c: np.linalg.norm(centroid[c] - centroid[ts]))
                pick = np.isin(sn, order[:k])
                m = train_mlp(preds[pick], gts[pick], dev, int(p["epochs"]), seed=base_seed)
                near_by_seq[ts] = float(eval_mlp(m, preds[te], gts[te], dev).mean())

                # fold 마다 독립 난수 — 원본은 모든 fold 가 같은 난수열을 썼다
                rng = np.random.default_rng([base_seed, bi, ki, abs(hash(ts)) % (2 ** 31)])
                trials = []
                for t in range(n_trials):
                    rpick = np.isin(sn, rng.choice(cands, k, replace=False))
                    m = train_mlp(preds[rpick], gts[rpick], dev, int(p["epochs"]), seed=base_seed + t)
                    trials.append(float(eval_mlp(m, preds[te], gts[te], dev).mean()))
                rand_by_seq[ts] = np.array(trials)

            if skip:
                results[bb]["k"][k] = {"note": "k >= 후보 수 — 오라클과 동일", "nearest_mm": oracle}
                print(f"  k={k:2d}  (전체 사용 = 오라클)", flush=True)
                continue

            s = summarize(near_by_seq, rand_by_seq, baseline, oracle, nframes)
            s["per_trial_random_mm"] = np.stack(
                [rand_by_seq[q] for q in sorted(rand_by_seq)]).mean(0).tolist()
            results[bb]["k"][k] = s
            print(f"      [프레임가중] nearest {s['fw_nearest_mm']:.3f} "
                  f"(오라클의 {s['fw_nearest_oracle_frac_pct']:.1f}%)  random {s['fw_random_mm']:.3f} "
                  f"±{s['fw_random_sd_mm']:.3f} (오라클의 {s['fw_random_oracle_frac_pct']:.1f}%)", flush=True)
            print(f"      @{bb}_k{k}_fw_nearest={s['fw_nearest_mm']:.4f} "
                  f"@{bb}_k{k}_fw_random={s['fw_random_mm']:.4f}", flush=True)
            print(f"  k={k:2d}  nearest {s['nearest_mm']:.3f} ({s['nearest_imp_pct']:.2f}%, "
                  f"오라클의 {s['nearest_oracle_frac_pct']:.1f}%)  |  random {s['random_mm']:.3f} "
                  f"±{s['random_sd_mm']:.3f} CI[{s['random_ci95_mm'][0]:.3f}, {s['random_ci95_mm'][1]:.3f}] "
                  f"({s['random_imp_pct']:.2f}%)  |  Wilcoxon p={s['wilcoxon_p']:.2e}  "
                  f"nearest 우세 fold {s['folds_nearest_better']}/{s['n_folds']}  "
                  f"random 승률 {s['random_draws_beating_nearest_pct']:.1f}%  "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
            print(f"  @{bb}_k{k}_nearest={s['nearest_mm']:.4f} @{bb}_k{k}_random={s['random_mm']:.4f} "
                  f"@{bb}_k{k}_random_sd={s['random_sd_mm']:.4f} @{bb}_k{k}_wilcoxon_p={s['wilcoxon_p']:.3e}",
                  flush=True)
            out = Path(p["out"])                       # 중간 저장 — 중단돼도 남는다
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(results, indent=2))

    print(f"\n저장: {p['out']}  (총 {(time.time() - t0) / 60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
