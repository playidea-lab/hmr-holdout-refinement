"""공식 HMR2 test npz 의 scale 은 (1.2·w, 1.2·h) 두 축 — 저장된 npz 의 scale 을 그 규약으로 고친다.

test 에서 고친 npz 가 공식 npz 와 같은 center/scale 을 갖는지 다시 확인한다.
"""
import os
import glob
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

SEQ = os.environ.get("THREEDPW_SEQ_DIR", "data/3DPW/sequenceFiles")
OUT = Path(os.path.join(os.environ.get("RESULTS_DIR", "outputs"), "clean_split"))
H = np.load(os.path.join(os.environ.get("HMR2_ROOT", "third_party/4D-Humans"), "hmr2_evaluation_data/3dpw_test.npz"), allow_pickle=True)


def wh(split):
    out = []
    for p in sorted(glob.glob(f"{SEQ}/{split}/*.pkl")):
        with open(p, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        for pid in range(len(d["poses"])):
            for fi in np.nonzero(np.asarray(d["campose_valid"][pid]).astype(bool))[0]:
                kp = np.asarray(d["poses2d"][pid][fi]).T
                v = kp[kp[:, 2] > 0]
                if len(v) == 0:
                    continue
                out.append(v[:, :2].max(0) - v[:, :2].min(0))
    return np.array(out)


for split in ("test", "train", "validation"):
    f = OUT / f"3dpw_{split}_seqfmt.npz"
    d = dict(np.load(f, allow_pickle=True))
    w = wh(split)
    assert len(w) == len(d["imgname"])
    d["scale"] = (1.2 * w).astype(np.float64)
    np.savez(f, **d)
    print(f"[{split}] scale 교정 {len(w)}행", flush=True)

d = np.load(OUT / "3dpw_test_seqfmt.npz", allow_pickle=True)
by = defaultdict(list)
for i, n in enumerate(H["imgname"]):
    by[str(n)].append(i)
m = np.array([min(by[n], key=lambda k: np.linalg.norm(H["center"][k] - c)) for n, c in zip(d["imgname"], d["center"])])
dc = np.abs(H["center"][m] - d["center"]).max()
ds = np.abs(H["scale"][m] - d["scale"]).max()
print(f"[검증] 공식 test 대비 center 최대차 {dc:.4f}px, scale 최대차 {ds:.4f} @scale_maxdiff={ds:.5f}", flush=True)
