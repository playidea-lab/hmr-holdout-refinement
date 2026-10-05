#!/usr/bin/env python3
"""exp770: 같은 라벨 예산을 백본 헤드에 썼을 때와 refiner에 썼을 때를 비교한다.

논문은 "nearest 5개 시퀀스를 라벨링해 refiner를 학습하라"고 권고하면서, 같은
라벨을 모델 자체에 썼을 때와 비교하지 않는다. 리뷰어 둘이 이걸 첫 번째로 물었다.

설계:
  - ViT-H 백본(630.9M)은 동결, SMPL 헤드(39.5M)만 학습한다. 16GB GPU에서
    전체 파인튜닝은 불가능하고, 헤드 학습이 "라벨을 모델에 쓴다"의 현실적 대응물이다.
  - 백본이 동결이므로 conditioning feats를 한 번 캐싱한다(35,515 x 1280x16x12,
    fp16 17.5GB). 이후 폴드별 학습은 캐시에서 돌아 refiner 실험만큼 싸다.
  - 예산·프로토콜은 refiner와 동일: 폴드마다 nearest 5시퀀스로 학습, held-out
    시퀀스에서 평가, pelvis 중심화 후 H36M-14 관절의 PA-MPJPE.

stage=cache 로 특징을 굽고, stage=train 으로 폴드를 돈다.
"""

import argparse
import functools
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# PyTorch 2.6의 weights_only 기본값이 이 체크포인트를 막는다
_orig_load = torch.load
torch.load = functools.partial(_orig_load, weights_only=False)

sys.path.insert(0, os.environ.get("HMR2_ROOT", "third_party/4D-Humans"))
# hmr2의 데이터셋 설정이 상대경로(hmr2_evaluation_data/...)를 쓴다
import os
os.chdir(os.environ.get("HMR2_ROOT", "third_party/4D-Humans"))

FEAT_SHAPE = (1280, 16, 12)
N_FRAMES = 35515
ANN = os.environ.get("TOKENHMR_3DPW_NPZ", "data/tokenhmr/3dpw_test.npz")
REFINER_PKL = os.path.join(os.environ.get("PRED_DIR", "data"), "4dhumans_3dpw_cuda.pkl")


def build_model():
    from hmr2.models import load_hmr2, download_models, DEFAULT_CHECKPOINT
    from hmr2.configs import CACHE_DIR_4DHUMANS
    download_models(CACHE_DIR_4DHUMANS)
    model, cfg = load_hmr2(DEFAULT_CHECKPOINT)
    return model.cuda().eval(), cfg


def build_loader(cfg, batch_size=32):
    from hmr2.configs import dataset_eval_config
    from hmr2.datasets import create_dataset
    dcfg = dataset_eval_config()["3DPW-TEST"]
    ds = create_dataset(cfg, dcfg, train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                     num_workers=4)
    return ds, dl, dcfg


def stage_cache(out_dir: Path) -> None:
    """conditioning feats와 GT를 한 번 구워 둔다."""
    from hmr2.utils import recursive_to
    model, cfg = build_model()
    ds, dl, dcfg = build_loader(cfg)
    n = len(ds)
    print(f"dataset {n} frames, feat {FEAT_SHAPE}", flush=True)

    feats = np.lib.format.open_memmap(out_dir / "feats.npy", mode="w+",
                                      dtype=np.float16, shape=(n, *FEAT_SHAPE))
    gt = np.zeros((n, 44, 3), dtype=np.float32)
    i = 0
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            batch = recursive_to(batch, torch.device("cuda"))
            x = batch["img"]
            f = model.backbone(x[:, :, :, 32:-32])
            b = f.shape[0]
            feats[i:i + b] = f.half().cpu().numpy()
            gt[i:i + b] = batch["keypoints_3d"][:, :, :-1].cpu().numpy()
            i += b
            if bi % 100 == 0:
                print(f"  {i}/{n}", flush=True)
    feats.flush()
    np.save(out_dir / "gt44.npy", gt)
    meta = {"n": int(n), "pelvis_ind": int(cfg.EXTRA.PELVIS_IND),
            "keypoint_list": [int(v) for v in dcfg.KEYPOINT_LIST]}
    (out_dir / "meta.json").write_text(json.dumps(meta))
    print(f"cached {i} frames -> {out_dir}", flush=True)


def similarity_transform(S1, S2):
    B = S1.shape[0]
    S1, S2 = S1.permute(0, 2, 1), S2.permute(0, 2, 1)
    mu1, mu2 = S1.mean(2, keepdim=True), S2.mean(2, keepdim=True)
    X1, X2 = S1 - mu1, S2 - mu2
    var1 = (X1 ** 2).sum(dim=(1, 2))
    K = torch.bmm(X1, X2.permute(0, 2, 1))
    U, _, Vh = torch.linalg.svd(K)
    Z = torch.eye(3, device=S1.device).unsqueeze(0).repeat(B, 1, 1)
    Z[:, -1, -1] = torch.sign(torch.linalg.det(torch.bmm(U, Vh)))
    R = torch.bmm(torch.bmm(Vh.permute(0, 2, 1), Z), U.permute(0, 2, 1))
    tr = torch.bmm(R, K).diagonal(dim1=-2, dim2=-1).sum(-1)
    sc = (tr / var1).unsqueeze(-1).unsqueeze(-1)
    return (sc * torch.bmm(R, S1) + (mu2 - sc * torch.bmm(R, mu1))).permute(0, 2, 1)


def pa_mpjpe(p, g):
    return torch.sqrt(((similarity_transform(p, g) - g) ** 2).sum(-1)).mean(-1) * 1000


def head_forward(model, feats):
    """캐시된 feats -> pred_keypoints_3d (B, 44, 3). 헤드와 SMPL만 통과."""
    params, cam, _ = model.smpl_head(feats)
    B = feats.shape[0]
    params["global_orient"] = params["global_orient"].reshape(B, -1, 3, 3)
    params["body_pose"] = params["body_pose"].reshape(B, -1, 3, 3)
    params["betas"] = params["betas"].reshape(B, -1)
    out = model.smpl(**{k: v.float() for k, v in params.items()}, pose2rot=False)
    return out.joints.reshape(B, -1, 3)


def stage_train(out_dir: Path, k: int, epochs: int, lr: float) -> None:
    import copy
    model, cfg = build_model()
    for p in model.backbone.parameters():
        p.requires_grad_(False)

    meta = json.loads((out_dir / "meta.json").read_text())
    pelvis, kl = meta["pelvis_ind"], meta["keypoint_list"]
    feats = np.load(out_dir / "feats.npy", mmap_mode="r")
    gt44 = np.load(out_dir / "gt44.npy")
    n = meta["n"]

    imgnames = np.load(ANN, allow_pickle=True)["imgname"][:n]
    tmp = defaultdict(list)
    for i, nm in enumerate(imgnames):
        tmp[nm.split("/")[1]].append(i)
    seqs = sorted(tmp)
    idx = {s: np.array(tmp[s]) for s in seqs}

    # refiner와 같은 선택 규칙: 시퀀스 평균 예측 포즈의 L2 거리
    with open(REFINER_PKL, "rb") as f:
        samp = pickle.load(f)
    pred_eval = np.stack([x["pred_joints_3d_eval"] for x in samp])[:n]
    mean_pred = {s: pred_eval[idx[s]].reshape(len(idx[s]), -1).mean(0) for s in seqs}

    init_head = copy.deepcopy(model.smpl_head.state_dict())
    before_all, after_all = [], []

    for s in seqs:
        cands = [q for q in seqs if q != s]
        d = np.array([np.linalg.norm(mean_pred[s] - mean_pred[q]) for q in cands])
        pick = [cands[i] for i in np.argsort(d)[:k]]
        tr_i = np.concatenate([idx[q] for q in pick])
        te_i = idx[s]

        model.smpl_head.load_state_dict(init_head)
        model.smpl_head.train()
        opt = torch.optim.AdamW(model.smpl_head.parameters(), lr=lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        for _ in range(epochs):
            perm = np.random.permutation(len(tr_i))
            for b0 in range(0, len(perm), 16):
                bi = np.sort(tr_i[perm[b0:b0 + 16]])
                f = torch.from_numpy(np.asarray(feats[bi])).float().cuda()
                g = torch.from_numpy(gt44[bi]).float().cuda()
                p3 = head_forward(model, f)
                p3 = p3 - p3[:, [pelvis]]
                g3 = g - g[:, [pelvis]]
                loss = torch.nn.functional.l1_loss(p3[:, kl], g3[:, kl])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.smpl_head.parameters(), 1.0)
                opt.step()
            sch.step()

        # 파인튜닝된 헤드로 전 배치를 평가한 뒤 한 번만 초기값으로 되돌려 baseline을
        # 잰다. 배치 안에서 되돌리면 다음 배치가 원본 헤드로 계산돼 둘이 같아진다.
        ft_head = copy.deepcopy(model.smpl_head.state_dict())
        model.smpl_head.eval()
        fold_after, fold_before = [], []
        with torch.no_grad():
            for state, sink in ((ft_head, fold_after), (init_head, fold_before)):
                model.smpl_head.load_state_dict(state)
                for b0 in range(0, len(te_i), 32):
                    bi = te_i[b0:b0 + 32]
                    f = torch.from_numpy(np.asarray(feats[bi])).float().cuda()
                    g = torch.from_numpy(gt44[bi]).float().cuda()
                    g3 = (g - g[:, [pelvis]])[:, kl]
                    p = head_forward(model, f)
                    p = (p - p[:, [pelvis]])[:, kl]
                    sink += pa_mpjpe(p, g3).cpu().tolist()
        after_all += fold_after
        before_all += fold_before
        print(f"  {s:<28s} n={len(te_i):5d}  "
              f"before {np.mean(fold_before):6.2f} -> "
              f"after {np.mean(fold_after):6.2f}", flush=True)

    b, a = float(np.mean(before_all)), float(np.mean(after_all))
    res = {"k": k, "epochs": epochs, "lr": lr, "before": b, "after": a,
           "pct": 100 * (b - a) / b}
    (out_dir / "results.json").write_text(json.dumps(res, indent=2))
    print(f"\nhead fine-tune k={k}: {b:.3f} -> {a:.3f} ({res['pct']:+.2f}%)  "
          f"@head_ft_k{k}_pa={a:.4f} @baseline_pa={b:.4f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cache", "train"], required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "cache":
        stage_cache(out)
    else:
        stage_train(out, args.k, args.epochs, args.lr)


if __name__ == "__main__":
    main()
