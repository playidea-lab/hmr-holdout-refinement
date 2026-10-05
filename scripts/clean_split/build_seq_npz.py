"""3DPW sequenceFiles 에서 train/validation/test 를 같은 규약의 HMR2 평가 npz 로 만든다.

- 중심: poses2d(openpose) 유효점 bbox 중심 — 공식 HMR2 test npz 와 0px 일치(확인됨).
- scale: 공식 test npz 의 scale 이 bbox 의 어떤 함수인지 test 에서 맞춘 뒤 같은 식을 적용한다.
- GT: 성별 SMPL(betas, poses, trans) 월드 정점 → cam_poses 로 카메라 좌표 → SMPL_to_J19.
  test 에서 공식 extra_keypoints_3d 를 재현하는지 먼저 검증한다.
"""
import os
import glob
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import smplx
import torch

SEQ = os.environ.get("THREEDPW_SEQ_DIR", "data/3DPW/sequenceFiles")
OUT = Path(os.path.join(os.environ.get("RESULTS_DIR", "outputs"), "clean_split"))
OUT.mkdir(parents=True, exist_ok=True)
SMPL_DIR = os.environ.get("SMPL_DIR", "data/body_models/smpl")
J19 = os.path.expanduser("~/.cache/4DHumans/data/SMPL_to_J19.pkl")
OFFICIAL = os.path.join(os.environ.get("HMR2_ROOT", "third_party/4D-Humans"), "hmr2_evaluation_data/3dpw_test.npz")
PELVIS = 14
BS = 512

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(J19, "rb") as f:
    JREG = torch.tensor(np.asarray(pickle.load(f, encoding="latin1")), dtype=torch.float32, device=dev)
MODELS = {g: smplx.SMPL(SMPL_DIR, gender={"m": "male", "f": "female"}[g]).to(dev) for g in "mf"}


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f, encoding="latin1")


def rows_of(split):
    R = defaultdict(list)
    for p in sorted(glob.glob(f"{SEQ}/{split}/*.pkl")):
        d = load(p)
        seq = d["sequence"]
        for pid in range(len(d["poses"])):
            g = d["genders"][pid]
            for fi in np.nonzero(np.asarray(d["campose_valid"][pid]).astype(bool))[0]:
                kp = np.asarray(d["poses2d"][pid][fi]).T
                v = kp[kp[:, 2] > 0]
                if len(v) == 0:
                    continue
                x0, y0 = v[:, :2].min(0)
                x1, y1 = v[:, :2].max(0)
                R["imgname"].append(f"imageFiles/{seq}/image_{fi:05d}.jpg")
                R["seq"].append(seq)
                R["pid"].append(pid)
                R["frame"].append(fi)
                R["center"].append([(x0 + x1) / 2, (y0 + y1) / 2])
                R["wh"].append([x1 - x0, y1 - y0])
                R["gender"].append(g)
                R["betas"].append(np.asarray(d["betas"][pid])[:10])
                R["pose"].append(np.asarray(d["poses"][pid][fi]))
                R["trans"].append(np.asarray(d["trans"][pid][fi]))
                R["cam"].append(np.asarray(d["cam_poses"][fi]))
    return {k: np.array(v) for k, v in R.items()}


def gt_extra(R):
    n = len(R["imgname"])
    out = np.zeros((n, 19, 3), np.float32)
    for g in "mf":
        idx = np.nonzero(R["gender"] == g)[0]
        for s in range(0, len(idx), BS):
            b = idx[s:s + BS]
            p = torch.tensor(R["pose"][b], dtype=torch.float32, device=dev)
            be = torch.tensor(R["betas"][b], dtype=torch.float32, device=dev)
            tr = torch.tensor(R["trans"][b], dtype=torch.float32, device=dev)
            cam = torch.tensor(R["cam"][b], dtype=torch.float32, device=dev)
            with torch.no_grad():
                v = MODELS[g](global_orient=p[:, :3], body_pose=p[:, 3:], betas=be, transl=tr).vertices
                v = torch.einsum("bij,bvj->bvi", cam[:, :3, :3], v) + cam[:, None, :3, 3]
                out[b] = torch.einsum("jv,bvc->bjc", JREG, v).cpu().numpy()
    return out


def fit_scale(R, H):
    by = defaultdict(list)
    for i, nm in enumerate(H["imgname"]):
        by[str(nm)].append(i)
    match = np.array([min(by[nm], key=lambda k: np.linalg.norm(H["center"][k] - c))
                      for nm, c in zip(R["imgname"], R["center"])])
    s0, s1 = H["scale"][match, 0], H["scale"][match, 1]
    w, h = R["wh"][:, 0], R["wh"][:, 1]
    cands = {"w": w, "h": h, "max": np.maximum(w, h)}
    best = None
    for sn, s in (("s0", s0), ("s1", s1)):
        for cn, x in cands.items():
            r = s / np.maximum(x, 1e-6)
            k = float(np.median(r))
            spread = float(np.median(np.abs(r / k - 1)))
            print(f"  {sn}/{cn}: 배수 {k:.4f}, 상대 편차 중앙값 {spread * 100:.2f}%", flush=True)
            if best is None or spread < best[0]:
                best = (spread, sn, cn, k)
    print(f"  s0==s1 비율 {np.mean(np.isclose(s0, s1)):.3f}", flush=True)
    return best, match


def scale_from(R, best):
    _, _, cn, k = best
    w, h = R["wh"][:, 0], R["wh"][:, 1]
    return k * {"w": w, "h": h, "max": np.maximum(w, h)}[cn]


def save(split, R, ex, scale):
    n = len(R["imgname"])
    np.savez(OUT / f"3dpw_{split}_seqfmt.npz",
             imgname=R["imgname"], center=R["center"].astype(np.float64),
             scale=np.stack([scale, scale], 1).astype(np.float64),
             body_pose=R["pose"].astype(np.float64), has_body_pose=np.zeros(n),
             betas=R["betas"].astype(np.float64), has_betas=np.zeros(n),
             body_keypoints_2d=np.zeros((n, 25, 3)),
             extra_keypoints_3d=np.concatenate([ex, np.ones((n, 19, 1), np.float32)], -1),
             gender=R["gender"], seq=R["seq"], pid=R["pid"], frame=R["frame"])
    print(f"[{split}] {n}행 저장", flush=True)


def main():
    H = np.load(OFFICIAL, allow_pickle=True)
    Rt = rows_of("test")
    print(f"test 행 {len(Rt['imgname'])} (공식 {len(H['imgname'])})", flush=True)
    best, match = fit_scale(Rt, H)
    print(f"채택 scale: {best[1]} ≈ {best[3]:.4f} × {best[2]} (편차 {best[0] * 100:.2f}%) @scale_spread={best[0]:.5f}", flush=True)
    ex = gt_extra(Rt)
    gt = H["extra_keypoints_3d"][match, :, :3]
    err = np.linalg.norm((ex - ex[:, [PELVIS]]) - (gt - gt[:, [PELVIS]]), axis=-1).mean() * 1000
    print(f"[검증] sequenceFile GT 대 공식 GT {err:.3f} mm @gt_err_mm={err:.4f}", flush=True)
    if err > 1.0:
        print("GT 재현 실패 — 중단", flush=True)
        sys.exit(1)
    save("test", Rt, ex, scale_from(Rt, best))
    for split in ("train", "validation"):
        R = rows_of(split)
        save(split, R, gt_extra(R), scale_from(R, best))
    print("@build_done=1", flush=True)


if __name__ == "__main__":
    main()
