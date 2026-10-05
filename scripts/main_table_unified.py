"""논문 B tab:main 을 tab:efficiency 와 같은 추정기로 다시 측정한다.

**왜**: 원고에 추정기가 두 개 섞여 있다. tab:main 의 Before 열(49.1/54.3/57.7)은 백본 pkl 에
저장된 pa_mpjpe 필드이고, tab:efficiency 와 우리 파이프라인은 관절에서 재계산한다
(49.5097/54.9427/58.1761). 백본마다 0.39~0.61 mm 의 계통 차이이고, 이 값이 논문의 모든
'% 개선'과 '% 오라클'의 분모다. 두 표가 같은 분모를 쓰게 만든다.

측정 규약은 exp803b 를 그대로 재사용한다(RefineMLP·compute_pa_mpjpe·train_mlp 동일 모듈에서 import).
오라클 = 24 시퀀스 LOO, 각 fold 는 나머지 23 시퀀스로 학습. 시드 3개의 평균과 표준편차를 낸다.
"""

import os
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from exp803b_data_efficiency_stats import (  # noqa: E402
    BACKBONES, DIM, MM, MOCAP, N_COORD, N_JOINTS,
    compute_pa_mpjpe, eval_mlp, load_backbone, train_mlp,
)

SEEDS = (0, 1, 2)
OUT = os.path.join(os.environ.get("RESULTS_DIR", "outputs"), "main_table_unified.json")


def compute_mpjpe(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """정렬 없는 관절 평균 거리(mm). 원본 exp700 compute_mpjpe 와 동일."""
    p = pred.reshape(-1, N_JOINTS, N_COORD)
    g = gt.reshape(-1, N_JOINTS, N_COORD)
    return np.sqrt(((p - g) ** 2).sum(-1)).mean(-1) * MM


@torch.no_grad()
def refine(model, preds, dev) -> np.ndarray:
    model.eval()
    x = torch.tensor(preds.reshape(-1, DIM), dtype=torch.float32, device=dev)
    return model(x).cpu().numpy().reshape(-1, N_JOINTS, N_COORD)


def loo_oracle(preds, gts, sn, seqs, dev, seed: int):
    """24 시퀀스 LOO. 각 held-out 시퀀스의 정제 결과를 원래 프레임 순서로 채워 돌려준다."""
    refined = np.empty_like(preds)
    for s in seqs:
        te = sn == s
        tr = ~te
        model = train_mlp(preds[tr], gts[tr], dev, seed=seed)
        refined[te] = refine(model, preds[te], dev)
    return refined


def main() -> int:
    p = dict(json.loads(__import__("os").environ.get("CQ_PARAMS", "{}")))
    seeds = tuple(int(x) for x in str(p.get("seeds", "0,1,2")).split(","))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev} seeds={seeds}", flush=True)

    mocap = np.load(MOCAP, allow_pickle=True)
    imgnames = [s.decode() if isinstance(s, bytes) else s for s in mocap["imgname"]]
    seq_names = [Path(n).parts[-2] if len(Path(n).parts) >= 2 else "unknown" for n in imgnames]

    out = {}
    t0 = time.time()
    for bb, path in BACKBONES.items():
        preds, gts, sn = load_backbone(path, seq_names)
        seqs = sorted(set(sn.tolist()))
        pa_b = compute_pa_mpjpe(preds, gts)
        mj_b = compute_mpjpe(preds, gts)
        print(f"\n===== {bb}  n={len(preds)} 시퀀스 {len(seqs)}", flush=True)
        print(f"  before  PA {pa_b.mean():.4f}  MPJPE {mj_b.mean():.4f}", flush=True)

        pa_after_seeds, mj_after_seeds, noworse_seeds, wp_seeds = [], [], [], []
        for seed in seeds:
            refined = loo_oracle(preds, gts, sn, seqs, dev, seed)
            pa_a = compute_pa_mpjpe(refined, gts)
            mj_a = compute_mpjpe(refined, gts)
            # 시퀀스 단위 대응 검정 — 표의 'No-worse' 열과 p 값
            sb = np.array([pa_b[sn == s].mean() for s in seqs])
            sa = np.array([pa_a[sn == s].mean() for s in seqs])
            noworse = int((sa <= sb).sum())
            try:
                _, wp = stats.wilcoxon(sb, sa)
            except ValueError:
                wp = 1.0
            pa_after_seeds.append(pa_a.mean())
            mj_after_seeds.append(mj_a.mean())
            noworse_seeds.append(noworse)
            wp_seeds.append(float(wp))
            print(f"  seed {seed}: PA {pa_a.mean():.4f}  MPJPE {mj_a.mean():.4f}  "
                  f"no-worse {noworse}/{len(seqs)}  wilcoxon p={wp:.3e}  "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)

        pa_a_m, pa_a_s = float(np.mean(pa_after_seeds)), float(np.std(pa_after_seeds, ddof=1))
        mj_a_m, mj_a_s = float(np.mean(mj_after_seeds)), float(np.std(mj_after_seeds, ddof=1))
        pa_imp = 100.0 * (pa_b.mean() - pa_a_m) / pa_b.mean()
        mj_imp = 100.0 * (mj_b.mean() - mj_a_m) / mj_b.mean()
        out[bb] = {
            "pa_before_mm": float(pa_b.mean()), "mpjpe_before_mm": float(mj_b.mean()),
            "pa_after_mm": pa_a_m, "pa_after_sd_mm": pa_a_s,
            "mpjpe_after_mm": mj_a_m, "mpjpe_after_sd_mm": mj_a_s,
            "pa_imp_pct": float(pa_imp), "mpjpe_imp_pct": float(mj_imp),
            "noworse_by_seed": noworse_seeds, "wilcoxon_p_by_seed": wp_seeds,
            "n_seq": len(seqs), "seeds": list(seeds),
        }
        print(f"  => PA {pa_b.mean():.2f} -> {pa_a_m:.2f} (±{pa_a_s:.2f}) {pa_imp:+.1f}%  |  "
              f"MPJPE {mj_b.mean():.2f} -> {mj_a_m:.2f} (±{mj_a_s:.2f}) {mj_imp:+.1f}%", flush=True)
        print(f"  @{bb}_pa_before={pa_b.mean():.4f} @{bb}_pa_after={pa_a_m:.4f} "
              f"@{bb}_mpjpe_before={mj_b.mean():.4f} @{bb}_mpjpe_after={mj_a_m:.4f} "
              f"@{bb}_imp_pct={pa_imp:.4f} @{bb}_wilcoxon_p={min(wp_seeds):.3e}", flush=True)

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n저장: {OUT}  (총 {(time.time()-t0)/60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
