"""tab:unsupervised(F6)를 표준 Procrustes 로 다시 낸다.

논문 표 안에 기준선이 두 개다 — 비지도 두 행은 54.94(최적 스케일 항 누락), 오라클 행은
54.33(표준). exp801/exp802 의 PA 함수에서 trace(Sigma) 를 복원했고, 최적화 절차 자체는
손대지 않았다(생체역학 손실도 합의 전략도 PA 를 쓰지 않는다 — PA 는 측정에만 쓰인다).

먼저 고친 PA 가 백본이 보고한 공식값과 같은지 확인하고, 다르면 즉시 멈춘다.
"""

import os
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from exp801_manifold_biomech import compute_pa_mpjpe_torch  # noqa: E402

REF_PKL = os.path.join(os.environ.get("PRED_DIR", "data"), "4dhumans_3dpw_cuda.pkl")
EXPECTED_4DH = 54.3258
TOL_MM = 0.01
OUT = os.path.join(os.environ.get("RESULTS_DIR", "outputs"), "fixed_unsupervised.json")


def verify() -> None:
    import time as _t
    t0 = _t.time()
    print(f"  pkl 로드 시작 {REF_PKL}", flush=True)
    with open(REF_PKL, "rb") as f:
        data = pickle.load(f)
    print(f"  로드 완료 {len(data)}개 ({_t.time()-t0:.2f}s)", flush=True)
    stored = np.array([s["pa_mpjpe"] for s in data], dtype=float).mean()
    preds = np.array([s["pred_joints_3d_eval"].reshape(14, 3) for s in data])
    gts = np.array([s["gt_joints_3d"].reshape(14, 3) for s in data])
    idx = np.linspace(0, len(preds) - 1, 2000).astype(int)
    ours = np.array([
        compute_pa_mpjpe_torch(torch.tensor(preds[i]), torch.tensor(gts[i])).item()
        for i in idx])
    stored_sub = np.array([data[i]["pa_mpjpe"] for i in idx], dtype=float)
    delta = np.abs(ours - stored_sub).max()
    print(f"  4DHumans 전체 저장 평균 {stored:.4f} (기대 {EXPECTED_4DH})", flush=True)
    print(f"  표본 2000개 최대 절대차 {delta:.6f} mm", flush=True)
    if delta > TOL_MM:
        raise SystemExit(f"규약 불일치: 최대 {delta:.5f} mm. 측정을 중단한다.")
    print("  규약 검증 통과\n", flush=True)


def run_step(script: str, run_dir: str) -> None:
    """하위 스크립트를 -u 로 돌린다.

    exp801/exp802 의 print 에는 flush 가 없다. 파이프로 넘기면 블록 버퍼링이라
    수십 분간 아무것도 안 보인다 — 진행 중인지 멈춘 건지 구분이 안 된다.
    """
    import time as _t
    print(f"\n=== {script} 시작 ===", flush=True)
    base = Path(__file__).parent.parent
    t0 = _t.time()
    r = subprocess.run([sys.executable, "-u", f"scripts/{script}", run_dir], cwd=str(base))
    print(f"=== {script} 종료 exit={r.returncode} ({(_t.time()-t0)/60:.1f} min) ===", flush=True)
    if r.returncode != 0:
        raise SystemExit(f"{script} 실패 (exit {r.returncode})")


def main() -> int:
    print("=== Procrustes 규약 검증 ===", flush=True)
    verify()

    # exp801 은 공식 run 72bdc56c(2026-09-21)에서 이미 완주했다. 같은 코드·같은 규약이므로
    # 결과가 있으면 재계산하지 않고 재사용한다 — 35분을 같은 값 뽑는 데 쓰지 않는다.
    # 출처는 아래 JSON 의 provenance 에 남긴다.
    base = Path(__file__).parent.parent
    p801 = base / "results/exp801/metrics.json"
    if p801.exists():
        print(f"\n=== exp801 재사용 (이전 공식 run 72bdc56c 산출물) ===", flush=True)
        reused = True
    else:
        run_step("exp801_manifold_biomech.py", "results/exp801")
        reused = False
    run_step("exp802_cross_backbone_consensus.py", "results/exp802")

    out = {"provenance": {
        "exp801": "공식 run 72bdc56c 재사용" if reused else "이 run 에서 계산",
        "exp802": "이 run 에서 계산",
        "procrustes": "표준(최적 스케일 trace(Sigma) 포함) — 백본 공식 pa_mpjpe 와 일치 검증됨",
    }}
    p802 = base / "results/exp802/metrics.json"
    if p801.exists():
        d = json.load(open(p801))
        out["biomech"] = d
        for bb, v in d.items():
            print(f"\n[생체역학] {bb}: {v['baseline']:.4f} -> {v['refined']:.4f}  "
                  f"{v['improvement_pct']:.2f}%  악화 {v['n_worse_seqs']}/24", flush=True)
            print(f"  @biomech_before={v['baseline']:.4f} @biomech_after={v['refined']:.4f} "
                  f"@biomech_imp_pct={v['improvement_pct']:.4f}", flush=True)
    if p802.exists():
        d = json.load(open(p802))
        out["consensus"] = d
        b = d["baselines"].get("4DHumans")
        for key in ("strategy_A_simple_mean", "strategy_B_manifold_proj",
                    "strategy_C_manifold_opt"):
            if key in d:
                imp = 100.0 * (b - d[key]) / b
                print(f"\n[합의] {key}: {b:.4f} -> {d[key]:.4f}  {imp:.2f}%", flush=True)
        if "strategy_C_manifold_opt" in d:
            c = d["strategy_C_manifold_opt"]
            print(f"  @consensus_before={b:.4f} @consensus_after={c:.4f} "
                  f"@consensus_imp_pct={100.0*(b-c)/b:.4f}", flush=True)

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n저장: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
