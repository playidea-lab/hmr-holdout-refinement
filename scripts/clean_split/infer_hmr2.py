"""HMR 2.0 을 3DPW train/validation/test(sequenceFile 규약 npz)에 돌려 14관절 예측·GT 를 저장한다.

공식 추출 경로와 같다: pred_keypoints_3d → 골반 중심 → KEYPOINT_LIST.
shuffle=False 로 npz 행 순서를 유지해 예측과 주석이 같은 행에 놓인다.
"""
import os
import copy
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("HMR2_ROOT", "third_party/4D-Humans"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from hmr2.configs import CACHE_DIR_4DHUMANS, dataset_eval_config  # noqa: E402
from hmr2.datasets import create_dataset  # noqa: E402
from hmr2.models import DEFAULT_CHECKPOINT, download_models, load_hmr2  # noqa: E402
from hmr2.utils import recursive_to  # noqa: E402

OUT = Path(os.path.join(os.environ.get("RESULTS_DIR", "outputs"), "clean_split"))


def similarity(S1, S2):
    S1, S2 = S1.permute(0, 2, 1), S2.permute(0, 2, 1)
    mu1, mu2 = S1.mean(2, keepdim=True), S2.mean(2, keepdim=True)
    X1, X2 = S1 - mu1, S2 - mu2
    var1 = (X1 ** 2).sum((1, 2))
    K = torch.bmm(X1, X2.permute(0, 2, 1))
    U, s, V = torch.svd(K)
    Z = torch.eye(3, device=S1.device).unsqueeze(0).repeat(len(S1), 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.linalg.det(U @ V.permute(0, 2, 1)))
    R = V @ Z @ U.permute(0, 2, 1)
    tr = (R @ K).diagonal(dim1=-1, dim2=-2).sum(-1)
    sc = (tr / var1)[:, None, None]
    t = mu2 - sc * (R @ mu1)
    return (sc * (R @ S1) + t).permute(0, 2, 1)


def run(model, model_cfg, split, npz):
    cfg = copy.deepcopy(dataset_eval_config()["3DPW-TEST"])
    cfg.defrost()
    cfg.DATASET_FILE = npz
    cfg.freeze()
    ds = create_dataset(model_cfg, cfg, train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=8)
    kl, pel = list(cfg.KEYPOINT_LIST), model_cfg.EXTRA.PELVIS_IND
    P, G, E = [], [], []
    for b in dl:
        b = recursive_to(b, "cuda")
        with torch.no_grad():
            o = model(b)
        p = o["pred_keypoints_3d"]
        g = b["keypoints_3d"][:, :, :-1]
        p = (p - p[:, [pel]])[:, kl]
        g = (g - g[:, [pel]])[:, kl]
        E.append((torch.sqrt(((similarity(p, g) - g) ** 2).sum(-1)).mean(-1) * 1000).cpu().numpy())
        P.append(p.cpu().numpy())
        G.append(g.cpu().numpy())
    P, G, E = np.concatenate(P), np.concatenate(G), np.concatenate(E)
    np.savez(OUT / f"hmr2_{split}.npz", pred=P, gt=G, pa=E)
    print(f"[{split}] n={len(E)} PA-MPJPE {E.mean():.3f} mm @pa_{split}={E.mean():.4f}", flush=True)


def main():
    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    model = model.cuda().eval()
    for split in ("test", "train", "validation"):
        run(model, model_cfg, split, str(OUT / f"3dpw_{split}_seqfmt.npz"))
    print("@infer_done=1", flush=True)


if __name__ == "__main__":
    main()
