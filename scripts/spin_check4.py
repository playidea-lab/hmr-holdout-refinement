"""SPIN pkl 행 순서가 SPIN 저장소의 3dpw_test.npz 순서를 따르는가.

따른다면: SPIN npz imgname → (시퀀스, 프레임, 사람) 키로 TokenHMR npz 색인을 찾아 재정렬하면
SPIN GT 가 HMR2 GT 와 같은 프레임이 되고(최근접 일치율 상승), 시퀀스 경계에서 포즈가 튄다.
"""

import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp803b_data_efficiency_stats as H  # noqa: E402
import spin_check3 as C  # noqa: E402

SPIN_NPZ = os.path.join(os.environ.get("SPIN_ROOT", "third_party/SPIN"), "data/dataset_extras/3dpw_test.npz")


def key_of(name, center):
    seq = name.split("/")[-2]
    frame = name.split("/")[-1]
    return seq, frame, tuple(np.round(center, 0))


def main():
    sp_npz = np.load(SPIN_NPZ, allow_pickle=True)
    tk_npz = np.load(H.MOCAP, allow_pickle=True)
    print("SPIN npz keys:", list(sp_npz.keys()), flush=True)
    sp_names = [s.decode() if isinstance(s, bytes) else s for s in sp_npz["imgname"]]
    tk_names = [s.decode() if isinstance(s, bytes) else s for s in tk_npz["imgname"]]
    print(f"SPIN npz n={len(sp_names)}  예: {sp_names[:2]}", flush=True)
    print(f"Token npz n={len(tk_names)}  예: {tk_names[:2]}", flush=True)
    same_order = sum(a.split('/')[-2:] == b.split('/')[-2:] for a, b in zip(sp_names, tk_names))
    print(f"같은 자리에서 (시퀀스,프레임) 일치: {same_order}/{min(len(sp_names), len(tk_names))}", flush=True)

    # (시퀀스, 프레임, 중심) 키로 SPIN 행 → Token npz 색인
    tk_index = {}
    for i, (n, c) in enumerate(zip(tk_names, tk_npz["center"])):
        tk_index.setdefault(key_of(n, c), []).append(i)
    mapping, miss, amb = [], 0, 0
    for n, c in zip(sp_names, sp_npz["center"]):
        cand = tk_index.get(key_of(n, c))
        if not cand:
            # 중심이 반올림 차로 어긋나면 (시퀀스,프레임) 만으로, 사람 둘이면 중심 거리로
            seq, frame = n.split("/")[-2], n.split("/")[-1]
            cand = [i for i, m in enumerate(tk_names) if m.endswith(f"{seq}/{frame}")] if miss < 0 else None
            miss += 1
            mapping.append(-1)
            continue
        amb += len(cand) > 1
        mapping.append(cand[0])
    mapping = np.array(mapping)
    print(f"키 매칭: 실패 {miss}  모호 {amb}  고유 대상 {len(set(mapping[mapping >= 0].tolist()))}", flush=True)
    np.save(os.path.join(os.environ.get("RESULTS_DIR", "outputs"), "exp775_spin_row_to_tokennpz.npy"), mapping)

    names = tk_names
    seq_tk = np.array([n.split("/")[-2] for n in names])
    _, g_h2, _ = H.load_backbone(H.BACKBONES["4DHumans"], names)
    p_sp, g_sp, _ = H.load_backbone(H.BACKBONES["SPIN"], names)
    ok = mapping >= 0
    rng = np.random.default_rng(0)
    q_rows = np.sort(rng.choice(np.nonzero(ok)[0], C.N_QUERY, replace=False))
    ref = C.desc(g_h2[:, C.LSP12])
    for nm, mp in (("as-is", C.MAP_AS_IS), ("swapped", C.MAP_SWAP)):
        q_desc = C.desc(g_sp[q_rows][:, mp])
        hits = near = 0
        for r, qd in zip(q_rows, q_desc):
            j = int(np.linalg.norm(ref - qd, axis=1).argmin())
            hits += j == mapping[r]
            near += abs(j - mapping[r]) <= 3
        print(f"재정렬 후 SPIN GT ({nm}) 최근접 = 대응 프레임 {hits}/{len(q_rows)}  ±3 {near}/{len(q_rows)}",
              flush=True)
        # 대응 프레임끼리 이름 맞춘 12관절 PA 거리
        print(f"   대응 프레임 PA 거리 vs HMR2 GT: "
              f"{C.pa if hasattr(C, 'pa') else ''}", flush=True)
    seq_sp = np.array([n.split("/")[-2] for n in sp_names])
    dd = np.linalg.norm(np.diff(C.desc(g_sp[:, C.MAP_AS_IS]), axis=0), axis=1)
    b = seq_sp[1:] != seq_sp[:-1]
    print(f"SPIN GT, SPIN npz 시퀀스 경계 기준: 내부 {np.median(dd[~b]):.4f}  경계 {np.median(dd[b]):.4f}",
          flush=True)
    print(f"시퀀스 수 SPIN npz {len(set(seq_sp.tolist()))}  Token npz {len(set(seq_tk.tolist()))}", flush=True)
    # SPIN npz 순서로 본 각 시퀀스 프레임 수 vs Token
    from collections import Counter
    cs, ct = Counter(seq_sp.tolist()), Counter(seq_tk.tolist())
    diff = {s: (cs[s], ct.get(s)) for s in cs if cs[s] != ct.get(s)}
    print(f"시퀀스별 프레임 수가 다른 시퀀스: {diff}", flush=True)


if __name__ == "__main__":
    main()
