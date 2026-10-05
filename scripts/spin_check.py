"""SPIN 파이프라인 점검 — 435.8mm GT 오프셋과 pelvis 오차 0이 버그인가.

학습 없음. 세 백본 pkl 의 필드·관절 순서·GT 를 3DPW 주석과 프레임별로 대조한다.
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp803b_data_efficiency_stats as H  # noqa: E402

MM = 1000.0


def describe(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    s0 = data[0]
    print(f"\n== {Path(path).name}: n={len(data)} type={type(s0).__name__}", flush=True)
    if isinstance(s0, dict):
        for k, v in s0.items():
            a = np.asarray(v) if not isinstance(v, (str, bytes)) else v
            shp = getattr(a, "shape", None)
            print(f"   {k}: {type(v).__name__} shape={shp} "
                  f"{'' if shp is None or a.size > 6 else a}", flush=True)
    return data


def per_joint_pa(preds, gts):
    """표준 Procrustes(H.compute_pa_mpjpe 와 같은 규약)로 관절별 평균 오차."""
    errs = []
    for p, g in zip(preds, gts):
        mu_p, mu_g = p.mean(0), g.mean(0)
        pc, gc = p - mu_p, g - mu_g
        sp, sg = np.sqrt((pc ** 2).sum()), np.sqrt((gc ** 2).sum())
        pc, gc = pc / (sp + 1e-8), gc / (sg + 1e-8)
        U, S, Vt = np.linalg.svd(pc.T @ gc)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1, 1, d]) @ U.T
        scale = S[0] + S[1] + d * S[2]
        al = sg * scale * (pc @ R.T) + mu_g
        errs.append(np.sqrt(((al - g) ** 2).sum(-1)) * MM)
    return np.array(errs)


def pa_dist(a, b):
    """두 관절 집합 사이 Procrustes 거리(mm, 프레임 평균)."""
    return float(H.compute_pa_mpjpe(a, b).mean())


def main():
    mocap = np.load(H.MOCAP, allow_pickle=True)
    print("3DPW npz keys:", list(mocap.keys()), flush=True)
    names = [s.decode() if isinstance(s, bytes) else s for s in mocap["imgname"]]
    print("npz n =", len(names), flush=True)
    raw = {}
    for bb, path in H.BACKBONES.items():
        raw[bb] = describe(path)
    loaded = {bb: H.load_backbone(p, names)[:2] for bb, p in H.BACKBONES.items()}
    n = min(len(v[0]) for v in loaded.values())
    print(f"\n공통 n={n}", flush=True)

    for bb, (p, g) in loaded.items():
        pj = per_joint_pa(p[:n], g[:n])
        print(f"\n[{bb}] PA {pj.mean():.4f}  관절별: " +
              " ".join(f"{x:.2f}" for x in pj.mean(0)), flush=True)
        # 관절 하나가 늘 원점(0)인가, GT·pred 에서 상수인가
        for name, arr in (("pred", p[:n]), ("gt", g[:n])):
            norm = np.linalg.norm(arr, axis=-1).mean(0) * MM
            zero_j = [j for j in range(arr.shape[1]) if np.allclose(arr[:, j], 0, atol=1e-6)]
            print(f"   {name}: 관절 노름 평균(mm) " + " ".join(f"{x:.0f}" for x in norm) +
                  f"   항상 0인 관절 {zero_j}", flush=True)
        # 관절 간 중복(같은 좌표) 검사
        dup = [(i, j) for i in range(14) for j in range(i + 1, 14)
               if np.allclose(g[:n, i], g[:n, j], atol=1e-5)]
        print(f"   GT 중복 관절쌍 {dup}", flush=True)

    # GT 끼리 대조: 같은 프레임이면 PA 거리 작아야 한다
    bbs = list(loaded)
    for i in range(len(bbs)):
        for j in range(i + 1, len(bbs)):
            a, b = loaded[bbs[i]][1][:n], loaded[bbs[j]][1][:n]
            print(f"\nGT {bbs[i]} vs {bbs[j]}: PA 거리 {pa_dist(a, b):.2f} mm", flush=True)
            # 프레임 어긋남 탐지: 시프트별 거리
            sub = np.arange(0, n - 50, 97)
            for sh in (-2, -1, 0, 1, 2):
                idx = sub + sh
                ok = (idx >= 0) & (idx < n)
                d = pa_dist(a[sub[ok]], b[idx[ok]])
                print(f"   shift {sh:+d}: {d:.2f}", flush=True)

    # 3DPW npz 의 GT 관절과 대조(있으면)
    for key in ("S", "body_keypoints", "pose_3d", "joints3d"):
        if key in mocap:
            print(f"npz '{key}' shape {mocap[key].shape}", flush=True)
    sp_raw = raw["SPIN"][0]
    print("\nSPIN 첫 표본 GT 14x3:\n", np.asarray(sp_raw.get("gt_joints_3d")).reshape(-1, 3)[:14],
          flush=True)
    print("HMR2 첫 표본 GT 14x3:\n", np.asarray(raw["4DHumans"][0].get("gt_joints_3d")).reshape(-1, 3)[:14],
          flush=True)
    print("SPIN 첫 표본 pred 14x3:\n", np.asarray(sp_raw.get("pred_joints_3d_eval")).reshape(-1, 3)[:14],
          flush=True)


if __name__ == "__main__":
    main()
