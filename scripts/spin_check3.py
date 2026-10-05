"""SPIN 프레임 정렬 검사: SPIN GT 의 프레임 i 와 가장 닮은 HMR2 GT 프레임이 i 인가.

회전 불변 서술자(이름 맞춘 12관절의 쌍거리 66차원, 스케일 정규화)로 최근접 프레임을 찾는다.
같은 프레임이면 최근접 색인 = 자기 색인(또는 ±수 프레임). 아니면 SPIN 파일의 프레임 순서가
3DPW npz 와 다르다 = 정렬 버그. 대조로 TokenHMR GT 에도 같은 검사를 한다.
"""

import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp803b_data_efficiency_stats as H  # noqa: E402

LSP12 = list(range(12))
MAP_AS_IS = [3, 2, 1, 4, 5, 6, 10, 9, 8, 11, 12, 13]
MAP_SWAP = [6, 5, 4, 1, 2, 3, 13, 12, 11, 8, 9, 10]
IU = np.triu_indices(12, 1)
N_QUERY = 600


def desc(j12):
    d = np.linalg.norm(j12[:, :, None, :] - j12[:, None, :, :], axis=-1)[:, IU[0], IU[1]]
    return d / (d.mean(1, keepdims=True) + 1e-8)


def nn_report(name, q_desc, ref_desc, q_idx, seq):
    hits, near, sameseq = 0, 0, 0
    offs = []
    for qi, qd in zip(q_idx, q_desc):
        dist = np.linalg.norm(ref_desc - qd, axis=1)
        j = int(dist.argmin())
        offs.append(j - qi)
        hits += j == qi
        near += abs(j - qi) <= 3
        sameseq += seq[j] == seq[qi]
    offs = np.array(offs)
    print(f"{name}: 최근접 = 자기 프레임 {hits}/{len(q_idx)}  ±3 이내 {near}/{len(q_idx)}  "
          f"같은 시퀀스 {sameseq}/{len(q_idx)}  오프셋 중앙값 {int(np.median(offs))}", flush=True)
    return offs


def main():
    mocap = np.load(H.MOCAP, allow_pickle=True)
    names = [s.decode() if isinstance(s, bytes) else s for s in mocap["imgname"]]
    seq = np.array([n.split("/")[-2] for n in names])
    _, g_h2, _ = H.load_backbone(H.BACKBONES["4DHumans"], names)
    _, g_tk, _ = H.load_backbone(H.BACKBONES["TokenHMR"], names)
    _, g_sp, _ = H.load_backbone(H.BACKBONES["SPIN"], names)
    rng = np.random.default_rng(0)
    q = np.sort(rng.choice(len(g_h2), N_QUERY, replace=False))
    ref = desc(g_h2[:, LSP12])
    nn_report("TokenHMR GT → HMR2 GT", desc(g_tk[q][:, LSP12]), ref, q, seq)
    for nm, mp in (("as-is", MAP_AS_IS), ("swapped", MAP_SWAP)):
        nn_report(f"SPIN GT ({nm}) → HMR2 GT", desc(g_sp[q][:, mp]), ref, q, seq)
    # SPIN 자기 자신의 pred 와 GT 는 같은 프레임인가 (내부 일관성)
    p_sp, _, _ = H.load_backbone(H.BACKBONES["SPIN"], names)
    pa_self = H.compute_pa_mpjpe(p_sp[q], g_sp[q]).mean()
    perm = rng.permutation(q)
    pa_perm = H.compute_pa_mpjpe(p_sp[q], g_sp[perm]).mean()
    print(f"SPIN pred vs 같은 행 GT PA {pa_self:.2f} mm, 섞은 GT PA {pa_perm:.2f} mm", flush=True)
    # 3DPW npz 의 2D 키포인트(이미지 평면)로 GT 프레임을 판정할 수는 없으니, 시퀀스 경계에서
    # 포즈 연속성으로 본다: 인접 행 GT 의 변화량이 시퀀스 경계에서만 크면 SPIN 행 순서도 npz 와 같다
    for nm, g, mp in (("HMR2", g_h2, LSP12), ("SPIN", g_sp, MAP_AS_IS)):
        dd = np.linalg.norm(np.diff(desc(g[:, mp]), axis=0), axis=1)
        boundary = seq[1:] != seq[:-1]
        print(f"{nm}: 인접 행 서술자 변화 — 시퀀스 내부 중앙값 {np.median(dd[~boundary]):.4f}  "
              f"경계 중앙값 {np.median(dd[boundary]):.4f}", flush=True)


if __name__ == "__main__":
    main()
