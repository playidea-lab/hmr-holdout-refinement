"""HMR 2.0 오라클을 10에폭과 30에폭에서 한 프로세스 안에서 나란히 잰다.

**왜 한 프로세스인가**: 부모 run 에서 epochs 파라미터를 30으로 줬는데 산출물이 이전 run 과
md5 까지 같았다. 파일 경로·아티팩트 수집 어딘가에서 결과가 갈아끼워졌다는 뜻이라, 두 설정을
같은 실행 안에서 계산하고 둘 다 stdout 으로 찍는다. 파일을 믿지 않는다.

**무엇을 가리나**: 논문은 §Training 에 10에폭을 명시하는데 tab:main 의 HMR 2.0 오라클은
42.66 이다. 10에폭 재현값은 42.3990(3시드, sd 0.165). exp700 의 기본 에폭이 30이므로
30에폭이 42.66 을 내는지 본다. TokenHMR·SPIN 은 논문과 0.06mm 안에서 일치하므로 대조군으로 함께 잰다.
"""

import os
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from exp803b_data_efficiency_stats import (  # noqa: E402
    BACKBONES, DIM, MOCAP, N_COORD, N_JOINTS,
    compute_pa_mpjpe, load_backbone, train_mlp,
)

SEEDS = (0, 1, 2)
EPOCH_SETS = (10, 30)
# 논문 tab:main 의 After 열 (비교 대상)
PAPER = {"TokenHMR": 39.27, "4DHumans": 42.66, "SPIN": 49.25}


@torch.no_grad()
def refine(model, preds, dev) -> np.ndarray:
    model.eval()
    x = torch.tensor(preds.reshape(-1, DIM), dtype=torch.float32, device=dev)
    return model(x).cpu().numpy().reshape(-1, N_JOINTS, N_COORD)


def loo_oracle(preds, gts, sn, seqs, dev, seed: int, epochs: int) -> float:
    refined = np.empty_like(preds)
    for s in seqs:
        te = sn == s
        model = train_mlp(preds[~te], gts[~te], dev, epochs, seed=seed)
        refined[te] = refine(model, preds[te], dev)
    return float(compute_pa_mpjpe(refined, gts).mean())


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev} seeds={SEEDS} epoch_sets={EPOCH_SETS}", flush=True)

    mocap = np.load(MOCAP, allow_pickle=True)
    imgnames = [s.decode() if isinstance(s, bytes) else s for s in mocap["imgname"]]
    seq_names = [Path(n).parts[-2] if len(Path(n).parts) >= 2 else "unknown" for n in imgnames]

    out, t0 = {}, time.time()
    for bb, path in BACKBONES.items():
        preds, gts, sn = load_backbone(path, seq_names)
        seqs = sorted(set(sn.tolist()))
        base = float(compute_pa_mpjpe(preds, gts).mean())
        print(f"\n===== {bb}  baseline {base:.4f}  (논문 tab:main after {PAPER[bb]})", flush=True)
        row = {"baseline_mm": base, "paper_after_mm": PAPER[bb]}
        for ep in EPOCH_SETS:
            vals = [loo_oracle(preds, gts, sn, seqs, dev, sd, ep) for sd in SEEDS]
            m, s = float(np.mean(vals)), float(np.std(vals, ddof=1))
            imp = 100.0 * (base - m) / base
            gap = m - PAPER[bb]
            row[f"oracle_ep{ep}"] = m
            row[f"sd_ep{ep}"] = s
            row[f"imp_ep{ep}"] = imp
            print(f"  {ep:2d} epochs: {m:.4f} (sd {s:.4f}) 개선 {imp:.2f}%  "
                  f"논문과 차 {gap:+.4f} mm   시드별 {[round(v, 4) for v in vals]}  "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
            print(f"  @{bb}_oracle_ep{ep}={m:.4f} @{bb}_imp_ep{ep}={imp:.4f}", flush=True)
        out[bb] = row

    print("\n=== 요약: 논문 값에 더 가까운 설정 ===", flush=True)
    for bb, r in out.items():
        d10, d30 = abs(r["oracle_ep10"] - r["paper_after_mm"]), abs(r["oracle_ep30"] - r["paper_after_mm"])
        print(f"  {bb:10s} 논문 {r['paper_after_mm']:.2f}  10ep 차 {d10:.3f}  30ep 차 {d30:.3f}"
              f"  -> {'10에폭' if d10 < d30 else '30에폭'}", flush=True)

    p = Path(os.path.join(os.environ.get("RESULTS_DIR", "outputs"), "epochs_oracle_compare.json"))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\n저장: {p}  (총 {(time.time()-t0)/60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
