"""백본들의 오차 부분공간이 서로 겹치는가 — 주각(principal angle) 분석.

논문 B(§F2)는 백본마다 잔차의 15개 주성분이 분산의 93~94%를 담고, 그 기저가 **LOO fold 간**
코사인 0.93~0.96으로 안정적이라고 보고했다. 그러나 **백본 사이**에 기저가 겹치는지는 재지 않았다.
§F5는 이식이 4분의 1만 남는다고 관찰하고 "겹침이 무엇을 따르는지는 둘보다 넓은 풀이 필요하다"로
끝난다. 주각은 둘로도 잴 수 있다.

가르는 것:
  기저가 거의 정렬돼 있는데도 이식이 실패한다 → 문제는 목표 부분공간이 아니라 예측→잔차 사상
  기저가 어긋나 있다 → 이식 실패는 기하로 설명이 끝난다

잔차 규약은 exp800_pca_residual_analysis.py 와 동일: residual = gt - pred, 42차원(14관절×3),
평균 중심화 후 SVD, 기저는 Vt[:d].

CQ_PARAMS: dims (쉼표 목록, 기본 5,10,15,20), backbones (쉼표 목록으로 이름 지정 시 그것만)
"""

import json
import os
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

BACKBONES = {
    "TokenHMR": os.path.join(os.environ.get("PRED_DIR", "data"), "tokenhmr_3dpw_cuda.pkl"),
    "4DHumans": os.path.join(os.environ.get("PRED_DIR", "data"), "4dhumans_3dpw_cuda.pkl"),
    "SPIN": os.path.join(os.environ.get("PRED_DIR", "data"), "spin_3dpw_14j.pkl"),
}
DIMS_DEFAULT = (5, 10, 15, 20)
N_JOINTS, N_COORD = 14, 3
DIM = N_JOINTS * N_COORD


def load_residuals(pkl_path: str) -> np.ndarray:
    """exp800 과 같은 규약: residual = gt - pred, (N, 42)."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    preds = np.array([s["pred_joints_3d_eval"].reshape(DIM) for s in data])
    gts = np.array([s["gt_joints_3d"].reshape(DIM) for s in data])
    return gts - preds


def pca_basis(residuals: np.ndarray, d: int) -> np.ndarray:
    """평균 중심화 후 상위 d개 주성분. 반환 (d, 42), 행이 정규직교."""
    centered = residuals - residuals.mean(0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt[:d]


def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """두 부분공간 사이 주각(도). A, B 는 행이 정규직교인 (d, D)."""
    s = np.linalg.svd(A @ B.T, compute_uv=False)
    return np.rad2deg(np.arccos(np.clip(s, -1.0, 1.0)))


def random_baseline(d: int, n_trial: int = 200, seed: int = 0) -> dict:
    """무작위 d차원 부분공간 두 개의 주각 — '겹친다'의 기준선.

    42차원 안의 15차원 두 개는 우연히도 상당히 겹친다. 이 기준선 없이는
    주각이 작다는 것만으로 공유를 주장할 수 없다.
    """
    rng = np.random.default_rng(seed)
    means, firsts = [], []
    for _ in range(n_trial):
        a = np.linalg.qr(rng.standard_normal((DIM, d)))[0].T
        b = np.linalg.qr(rng.standard_normal((DIM, d)))[0].T
        ang = principal_angles(a, b)
        means.append(ang.mean())
        firsts.append(ang[0])
    return {"mean": float(np.mean(means)), "mean_sd": float(np.std(means)),
            "first": float(np.mean(firsts))}


def main() -> int:
    p = json.loads(os.environ.get("CQ_PARAMS") or "{}")
    dims = tuple(int(x) for x in str(p.get("dims", ",".join(map(str, DIMS_DEFAULT)))).split(","))
    names = [n.strip() for n in p["backbones"].split(",")] if "backbones" in p else list(BACKBONES)

    res = {}
    for n in names:
        path = BACKBONES[n]
        if not Path(path).exists():
            print(f"[!] 없음: {path}", flush=True)
            continue
        r = load_residuals(path)
        res[n] = r
        print(f"{n}: 잔차 {r.shape}, RMS {np.sqrt((r ** 2).sum(-1).mean()) * 1000:.1f} mm", flush=True)
    if len(res) < 2:
        print("백본이 둘 미만이라 비교 불가", flush=True)
        return 1

    for d in dims:
        bases = {n: pca_basis(r, d) for n, r in res.items()}
        base = random_baseline(d)
        print(f"\n===== d = {d} =====", flush=True)
        print(f"무작위 부분공간 기준선: 평균 주각 {base['mean']:.1f}° (±{base['mean_sd']:.1f}), "
              f"최소 주각 {base['first']:.1f}°", flush=True)
        print(f"{'백본 쌍':<24}{'평균 주각':>10}{'최소':>8}{'최대':>8}   주각 목록(도)", flush=True)
        for a, b in combinations(bases, 2):
            ang = principal_angles(bases[a], bases[b])
            lst = " ".join(f"{x:.0f}" for x in ang[:min(d, 8)])
            print(f"{a + ' vs ' + b:<24}{ang.mean():>9.1f}°{ang.min():>7.1f}°{ang.max():>7.1f}°   {lst}"
                  f"{' …' if d > 8 else ''}", flush=True)
            tag = f"{a}_{b}_d{d}".replace(" ", "")
            print(f"@{tag}_mean={ang.mean():.2f} @{tag}_min={ang.min():.2f} "
                  f"@{tag}_max={ang.max():.2f}", flush=True)
        print(f"@random_d{d}_mean={base['mean']:.2f}", flush=True)

    print("\n해석: 백본 쌍의 평균 주각이 무작위 기준선보다 뚜렷이 작으면 부분공간이 실제로 공유된다.", flush=True)
    print("      그런데도 이식이 4분의 1만 남는다면(논문 §F5), 한계는 목표 공간이 아니라 사상에 있다.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
