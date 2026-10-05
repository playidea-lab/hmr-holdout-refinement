"""올바른 Procrustes 규약으로 tab:main 과 tab:efficiency 를 한 번에 다시 낸다.

compute_pa_mpjpe 는 exp803b 모듈에서 고쳤고(최적 스케일 항 복원), main_table_unified 는
그 모듈에서 import 하므로 두 표가 자동으로 같은 규약을 쓴다.

먼저 규약이 실제로 고쳐졌는지 검증한다 — 저장된 공식 pa_mpjpe 와 일치하지 않으면 즉시 멈춘다.
측정이 틀린 채로 한 시간을 태우는 것보다 낫다.
"""

import os
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import exp803b_data_efficiency_stats as eff  # noqa: E402
import main_table_unified as mt  # noqa: E402

TOL_MM = 0.01
EXPECTED = {"TokenHMR": 49.1241, "4DHumans": 54.3258, "SPIN": 57.6480}


def verify_convention(seq_names) -> None:
    """고친 compute_pa_mpjpe 가 백본이 보고한 값과 같은지 확인한다."""
    for bb, path in eff.BACKBONES.items():
        with open(path, "rb") as f:
            raw = pickle.load(f)
        stored = np.array([s["pa_mpjpe"] for s in raw], dtype=float).mean()
        preds, gts, _ = eff.load_backbone(path, seq_names)
        ours = eff.compute_pa_mpjpe(preds, gts).mean()
        delta = abs(ours - stored)
        print(f"  검증 {bb:10s} 저장 {stored:.4f}  우리 {ours:.4f}  차 {delta:+.5f}", flush=True)
        if delta > TOL_MM:
            raise SystemExit(
                f"규약 불일치: {bb} 에서 {delta:.5f} mm 차 (허용 {TOL_MM}). 측정을 중단한다.")
        if abs(stored - EXPECTED[bb]) > TOL_MM:
            raise SystemExit(f"{bb} 저장값이 예상과 다르다 — 입력 파일이 바뀌었는지 확인할 것.")
    print("  규약 검증 통과 — 백본 공식값과 동일한 Procrustes\n", flush=True)


def main() -> int:
    mocap = np.load(eff.MOCAP, allow_pickle=True)
    imgnames = [s.decode() if isinstance(s, bytes) else s for s in mocap["imgname"]]
    seq_names = [Path(n).parts[-2] if len(Path(n).parts) >= 2 else "unknown" for n in imgnames]

    print("=== Procrustes 규약 검증 ===", flush=True)
    verify_convention(seq_names)

    print("=== tab:main (3 시드) ===", flush=True)
    mt.OUT = os.path.join(os.environ.get("RESULTS_DIR", "outputs"), "fixed_main_table.json")
    mt.main()

    print("\n=== tab:efficiency (20 시행) ===", flush=True)
    eff.main()
    src = Path(os.path.join(os.environ.get("RESULTS_DIR", "outputs"), "exp803b_data_efficiency_stats.json"))
    dst = Path(os.path.join(os.environ.get("RESULTS_DIR", "outputs"), "fixed_data_efficiency.json"))
    if src.exists():
        shutil.copy(src, dst)
        print(f"복사: {dst}", flush=True)

    # 두 표가 같은 값을 가리키는지 확인 — k=23 최근접은 정의상 오라클과 같아야 한다
    with open(os.path.join(os.environ.get("RESULTS_DIR", "outputs"), "fixed_main_table.json")) as f:
        mtab = json.load(f)
    with open(dst) as f:
        etab = json.load(f)
    print("\n=== 두 표 정합 (오라클 대 k=23) ===", flush=True)
    for bb in mtab:
        k23 = etab.get(bb, {}).get("k", {}).get("23", {}).get("nearest_mm")
        print(f"  {bb:10s} tab:main after {mtab[bb]['pa_after_mm']:.4f} "
              f"(시드 평균)  |  tab:efficiency k=23 {k23}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
