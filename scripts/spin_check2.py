"""SPIN 관절 순서 대조: SPIN GT 를 HMR2.0(LSP 순서) GT 와 이름 맞춰 비교.

LSP-14(HMR2/TokenHMR): r_ank r_kne r_hip l_hip l_kne l_ank r_wri r_elb r_sho l_sho l_elb l_wri neck head
SPIN 파일은 0번이 원점(골반)이다. 후보 매핑 두 가지(좌우 라벨 그대로 / 뒤바꿈)를 모두 재고,
각 경우 PA 거리를 본다. 같은 프레임의 같은 사람 GT 라면 수 mm 여야 한다.
"""

import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp803b_data_efficiency_stats as H  # noqa: E402

LSP12 = list(range(12))                          # 손목·팔꿈치·어깨·엉덩이·무릎·발목 12개
# SPIN 파일 순서(첫 표본 좌표로 추정): 0 골반, 1-3 한쪽 다리(엉덩이·무릎·발목), 4-6 반대쪽,
# 7 흉부/목, 8-10 한쪽 팔(어깨·팔꿈치·손목), 11-13 반대쪽 팔
MAP_AS_IS = [3, 2, 1, 4, 5, 6, 10, 9, 8, 11, 12, 13]     # 1-3·8-10 을 '오른쪽'으로
MAP_SWAP = [6, 5, 4, 1, 2, 3, 13, 12, 11, 8, 9, 10]      # 1-3·8-10 을 '왼쪽'으로


def pa(a, b):
    out = []
    for p, g in zip(a, b):
        mu_p, mu_g = p.mean(0), g.mean(0)
        pc, gc = p - mu_p, g - mu_g
        sp, sg = np.sqrt((pc ** 2).sum()), np.sqrt((gc ** 2).sum())
        pc, gc = pc / (sp + 1e-8), gc / (sg + 1e-8)
        U, S, Vt = np.linalg.svd(pc.T @ gc)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1, 1, d]) @ U.T
        sc = S[0] + S[1] + d * S[2]
        out.append(np.sqrt(((sg * sc * (pc @ R.T) + mu_g - g) ** 2).sum(-1)).mean() * 1000)
    return float(np.mean(out))


def main():
    mocap = np.load(H.MOCAP, allow_pickle=True)
    names = [s.decode() if isinstance(s, bytes) else s for s in mocap["imgname"]]
    _, g_h2, _ = H.load_backbone(H.BACKBONES["4DHumans"], names)
    _, g_tk, _ = H.load_backbone(H.BACKBONES["TokenHMR"], names)
    p_sp, g_sp, _ = H.load_backbone(H.BACKBONES["SPIN"], names)
    sub = np.arange(0, len(g_h2), 7)
    for nm, mp in (("as-is", MAP_AS_IS), ("swapped", MAP_SWAP)):
        d_h2 = pa(g_sp[sub][:, mp], g_h2[sub][:, LSP12])
        d_tk = pa(g_sp[sub][:, mp], g_tk[sub][:, LSP12])
        # x 뒤집기(거울)까지 허용하면?
        flip = g_sp[sub][:, mp] * np.array([-1, 1, 1])
        d_flip = pa(flip, g_h2[sub][:, LSP12])
        print(f"SPIN GT {nm:8s}: vs HMR2 GT {d_h2:7.2f} mm  vs TokenHMR GT {d_tk:7.2f} mm  "
              f"(x 반전 후 vs HMR2 {d_flip:7.2f})", flush=True)
    print(f"참고 HMR2 vs TokenHMR GT(12관절): {pa(g_h2[sub][:, LSP12], g_tk[sub][:, LSP12]):.2f} mm",
          flush=True)
    # SPIN 의 골반(0번) 오차와 개선 가능성: pred/gt 모두 0 이므로 잔차 0
    print(f"SPIN pred[:,0] 최대 |값| {np.abs(p_sp[:, 0]).max():.2e}  gt[:,0] 최대 {np.abs(g_sp[:, 0]).max():.2e}",
          flush=True)
    # 오른쪽/왼쪽 판정: 각 GT 에서 '오른쪽 엉덩이'의 x 부호 (카메라 좌표)
    print(f"HMR2 r_hip x 평균 {g_h2[:, 2, 0].mean():+.3f}  l_hip x {g_h2[:, 3, 0].mean():+.3f}", flush=True)
    print(f"SPIN j1 x 평균 {g_sp[:, 1, 0].mean():+.3f}  j4 x {g_sp[:, 4, 0].mean():+.3f}", flush=True)


if __name__ == "__main__":
    main()
